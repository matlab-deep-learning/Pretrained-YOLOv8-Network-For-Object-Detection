function [box, cls] = yolov8Transform(predictions,numClasses)
% Transforms predictions from dlnetwork to
% box [x_center y_center w h] and cls scores

regMax = 16;
outputsPerAnchor = numClasses + regMax*4;
stride = [8,16,32];
batch = 1;

pred = cell(3,1);
predPermute = cell(3,1);
predAll = cell(3,1);

for i = 1:size(predictions,1)
    pred{i,1} = permute(predictions{i,1},[2,1,3,4]);
    predPermute{i,1} = reshape(pred{i,1},[],outputsPerAnchor,batch);
    predAll{i,1} = permute(predPermute{i,1},[2,1,3,4]);
end

% Concat all Predictions
predCat = cat(2,predAll{:});

% Split classes and boxes
box = predCat(1:64,:,:);
cls = predCat(65:end,:,:);

% Compute anchor grid and stride
[anchorGrid, stride] = helper.make_anchors(predictions, stride);
% anchor grid and stride transposed
anchorGrid = anchorGrid';
stride = stride';

box = helper.distributionFocalLoss(box);
% Converting boxes to xywh format here
box = helper.dist2bbox(box,anchorGrid);
box = box .* stride;
% Sigmoid of classes
cls = sigmoid(dlarray(cls));
cls = extractdata(cls);
end
