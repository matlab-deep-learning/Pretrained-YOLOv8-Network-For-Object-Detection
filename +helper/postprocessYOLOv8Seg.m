% Apply postprocessing on the output feature maps.
function [masks,labelIds, scores, bboxes] = postprocessYOLOv8Seg(outFeatureMaps, ...
    origSize, newSize, numClasses)

% Reshape predictions
[detectionPriors, detectionDimension, maskPriors] = iReshapePredictions(outFeatureMaps);

% Compute box and class priors
[boxPriors, clsPriors] = iGetBoundingBoxesAndClasses(detectionPriors, detectionDimension);

% Obtain boxes, scores and labels
confThreshold = 0.5;
[bboxesNorm, scores, labelIds, fullDets]  = iExtractDetections(boxPriors, clsPriors, maskPriors, numClasses, confThreshold);

shape = newSize(1:2);
masks = iExtractMasks(outFeatureMaps{7,1}, fullDets(:,7:end), bboxesNorm, shape, origSize);
bboxesScaled = iScaleBoxes(shape, bboxesNorm, origSize);
bboxes = x1y1x2y2ToXYWH(bboxesScaled);
end

%--------------------------------------------------------------------------
function [detectionPriors, detectionDimension, maskPriors] = iReshapePredictions(fout)
% First three outputs correspond to mask priors
Z1Conv = fout{1,1};
batchSize = size(Z1Conv,4);
Z1Convmc = permute(Z1Conv,[2,1,3,4]);
Z1mc = reshape(Z1Convmc,[],32,batchSize);

Z2Conv = fout{2,1};
Z2Convmc = permute(Z2Conv,[2,1,3,4]);
Z2mc = reshape(Z2Convmc,[],32,batchSize);

Z3Conv = fout{3,1};
Z3Convmc = permute(Z3Conv,[2,1,3,4]);
Z3mc = reshape(Z3Convmc,[],32,batchSize);

maskPriors = cat(1, Z1mc, Z2mc, Z3mc);

% last 3 priors correspond to detection priors
detectionDimension = cell(1,3);
Z1x = fout{4,1};
Z1ViewxCat = permute(Z1x,[2,1,3,4]);
detectionDimension{1,1} = size(Z1ViewxCat);
Z1xCat = reshape(Z1ViewxCat,[],144,batchSize);

Z2x = fout{5,1};
Z2ViewxCat = permute(Z2x,[2,1,3,4]);
detectionDimension{1,2} = size(Z2ViewxCat);
Z2xCat = reshape(Z2ViewxCat,[],144,batchSize);

Z3x = fout{6,1};
Z3ViewxCat = permute(Z3x,[2,1,3,4]);
detectionDimension{1,3} = size(Z3ViewxCat);
Z3xCat = reshape(Z3ViewxCat,[],144,batchSize);

detectionPriors = cat(1,Z1xCat,Z2xCat,Z3xCat);
end

%--------------------------------------------------------------------------
function [boxPriors, clsPriors] = iGetBoundingBoxesAndClasses(detectionPriors,detectionDimension)

stride = [8, 16, 32];
anchorMap = cell(3,1);
anchorMap{1,1} = zeros(detectionDimension{1,1});
anchorMap{2,1} = zeros(detectionDimension{1,2});
anchorMap{3,1} = zeros(detectionDimension{1,3});
anchorGrid = computeSegmentationAnchors(anchorMap, stride);
box = detectionPriors(:,1:64);
cls = detectionPriors(:,65:end);

% Decode boxes
bboxData = reshape(box,[],16,4);

% Apply softmax operation
X = bboxData;
X = X - max(X,[],2);
X = exp(X);
softmaxOut = X./sum(X,2);

softmaxOut = permute(softmaxOut,[3,1,2]);
softmaxOut = dlarray(single(softmaxOut),'SSCB');

% Compute Distribution Focal Loss (DFL)
weights = dlarray(single(reshape(0:15, [1, 1, 16])));
bias = dlarray(single(0));

convOut = dlconv(softmaxOut, weights, bias);
convOut = extractdata(convOut);
convOut = permute(convOut,[2,1]);

% Transform distance (ltrb) to box (xywh)
lt = convOut(:,1:2);
rb = convOut(:,3:4);

x1y1 = anchorGrid - lt;
x2y2 = anchorGrid + rb;

% Compute centre
cxy = (x1y1 + x2y2)./2;

% Compute width and height values
wh = x2y2 - x1y1;

% bbox values
boxOut = cat(2,cxy,wh);

% dbox values
largestFeatureMapSize = detectionDimension{1,1}(1,1).*detectionDimension{1,1}(1,2);
mulConst = [8.*ones(largestFeatureMapSize,1);16.*ones(largestFeatureMapSize./4,1);32.*ones(largestFeatureMapSize./16,1)];

boxPriors = boxOut.* mulConst;
clsPriors = sigmoid(dlarray(cls));
end

%--------------------------------------------------------------------------
function [bboxes, scores, labelIds, fullDets] = iExtractDetections(boxPriors, clsPriors, maskPriors, numClasses, confThresh)
infOut = cat(2, boxPriors, clsPriors);
pred = cat(2, infOut, maskPriors);
maskIndex = 4 + numClasses;

tmpVal = pred(:, 5:maskIndex);
tmpMaxVal = max(tmpVal,[],2);
boxCandidates = tmpMaxVal > confThresh;

pred(:,1:4) = computeBoxes(pred(:,1:4));

predFull = extractdata(pred(boxCandidates, :));

box = predFull(:, 1:4);
cls = predFull(:, 5:5 + numClasses-1);
mask = predFull(:, 5 + numClasses:end);

[clsConf,ind] = max(cls,[],2);
fullDets = cat (2, box, clsConf, ind, mask);

fullDets = fullDets(clsConf > confThresh,:);

bboxesTmp = fullDets(:,1:4);
scoresTmp = fullDets(:, 5);

iou_thres = 0.8; % IoU threshold for NMS

% Apply NMS
[bboxes, scores, labelIds, idx] = selectStrongestBboxMulticlass(bboxesTmp, scoresTmp, ind, ...
    'RatioType', 'Min', 'OverlapThreshold', iou_thres);

fullDets = fullDets(idx,:);
end

%--------------------------------------------------------------------------
function [mask,downsampled_bboxes] = iExtractMasks(protoDL, masks_in, bboxes, shape, origShape)
[mh, mw, c] = size(protoDL);
[ih, iw] = deal(shape(1),shape(2));

proto = extractdata(protoDL);
protPermute = permute(proto,[3,2,1]);
protoVal = reshape(protPermute,c,[]);

maskTmp = masks_in*protoVal;

% Match Python code
maskTmpTrans = permute(maskTmp,[2,1]);
masks = reshape(maskTmpTrans,mw,mh,[]);
masks = permute(masks,[2,1,3]);

% Vectorized bbox calculations
scale = [mw./iw, mh./ih, mw./iw, mh./ih];
downsampled_bboxes = bboxes .* scale;

masks = iCropMasks(masks, downsampled_bboxes);

% Resize masks efficiently
mask = false([origShape(1:2), size(masks, 3)]);  % Preallocate as logical
for i = 1:size(masks, 3)
    mask(:,:,i) = imresize(masks(:,:,i), [origShape(1), origShape(2)], 'bilinear') > 0;
end
end

%--------------------------------------------------------------------------
function newCoords = iScaleBoxes(img1_shape,coords,img0_shape)
% Rescale coords (xyxy) from img1_shape to img0_shape
gain = min(img1_shape(1) / img0_shape(1), img1_shape(2) / img0_shape(2));  
pad = coder.nullcopy(zeros(1,2));
pad(1) = (img1_shape(2) - img0_shape(2) * gain)./ 2;
pad(2) = (img1_shape(1) - img0_shape(1) * gain)./ 2; 

newCoords = coords - repmat(pad,size(coords,1),2);
newCoords = newCoords./gain;
newCoords(newCoords<0) = 0.1;

newCoords(:,1) = min(newCoords(:,1),img0_shape(2));
newCoords(:,2) = min(newCoords(:,2),img0_shape(1));
newCoords(:,3) = min(newCoords(:,3),img0_shape(2));
newCoords(:,4) = min(newCoords(:,4),img0_shape(1));

end

%--------------------------------------------------------------------------
function boxes = x1y1x2y2ToXYWH(boxes)
% Convert [x1 y1 x2 y2] boxes into [x y w h] format. Input and
% output boxes are in pixel coordinates. boxes is an M-by-4
% matrix.
boxes(:,3) = boxes(:,3) - boxes(:,1) + 1;
boxes(:,4) = boxes(:,4) - boxes(:,2) + 1;
end


%--------------------------------------------------------------------------
function anchorGrid = computeSegmentationAnchors(feats, strideValues)

gridCellOffset = 0.5;
n = 3;
anchorGridTmp = cell(n,1);
totalSize = 0;

for i = 1:n
    sz = size(feats{i});
    totalSize = totalSize + (sz(1).*sz(2));
    anchorGridTmp{i,1} = coder.nullcopy(zeros(sz(1).*sz(2),2));
end

for i=1:size(strideValues,2)
    [h,w,~,~]= size(feats{i});
    sx = (0:h-1)+gridCellOffset;
    sy = (0:w-1)+gridCellOffset;
    [sy,sx]= meshgrid(sy,sx);
    anchorGridTmp{i,1} = cat(2, sx(:), sy(:));
end
anchorGrid = cat(1,anchorGridTmp{:});
end

%--------------------------------------------------------------------------
function boxCentres = computeBoxes(boxCandidatesDL)
boxCandidates = extractdata(boxCandidatesDL);
dw = boxCandidates(:,3)./2;
dh = boxCandidates(:,4)./2;

% Initialize y with the same size as x
boxCentres = zeros(size(boxCandidates));

% Calculate top left x and y
boxCentres(:, 1) = boxCandidates(:, 1) - dw;
boxCentres(:, 2) = boxCandidates(:, 2) - dh;

% Calculate bottom right x and y
boxCentres(:, 3) = boxCandidates(:, 1) + dw;
boxCentres(:, 4) = boxCandidates(:, 2) + dh;

end

%--------------------------------------------------------------------------
function resultMasks = iCropMasks(masks, boxes)
[rows, cols, numBoxes] = size(masks);
[r, c] = ndgrid(1:rows, 1:cols);
resultMasks = zeros(size(masks), 'like', masks);  % Use same data type as input

% Vectorized box coordinates
boxes = boxes + 1;  % Add 1 to all coordinates at once
for i = 1:numBoxes
    logicalMask = (r >= boxes(i,2)) & (r < boxes(i,4)) & ...
        (c >= boxes(i,1)) & (c < boxes(i,3));
    resultMasks(:,:,i) = masks(:,:,i) .* logicalMask;
end
end