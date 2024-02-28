function [box, cls] = yolov8Transform(predictions,numClasses)
% Transforms predictions from dlnetwork to
% box [x_center y_center w h] and cls scores

regMax = 16;
outputsPerAnchor = numClasses + regMax*4;
stride = [8,16,32];
batch = 1;

% Extract feature maps from Matlab model
% Apply this if dlarray output
predictions = cellfun(@extractdata,predictions,'UniformOutput',false);
predictions = cellfun(@gather,predictions,'UniformOutput',false);

% Compute anchor grid and stride
[anchorGrid, stride] = helper.make_anchors(predictions, stride);
% anchor grid and stride transposed
anchorGrid = anchorGrid';
stride = stride';

% Reshape predictions from model output
pred = cellfun(@(p){permute(p,[2,1,3,4])}, predictions, 'UniformOutput',true);
pred = cellfun(@(p){reshape(p,[],outputsPerAnchor, batch)}, pred, 'UniformOutput',true);
pred = cellfun(@(p){permute(p,[2,1,3,4])}, pred, 'UniformOutput',true);

% Concat all Predictions
predCat = cat(2,pred{:});

% Split classes and boxes
box = predCat(1:64,:,:);
cls = predCat(65:end,:,:);

box = helper.distributionFocalLoss(box);
% Converting boxes to xywh format here
box = helper.dist2bbox(box,anchorGrid);
box = box .* stride;
% Sigmoid of classes
cls = sigmoid(dlarray(cls));
cls = extractdata(cls);
end
