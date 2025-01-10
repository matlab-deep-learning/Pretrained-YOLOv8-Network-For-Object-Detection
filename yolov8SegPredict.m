function [masks,labelIds, scores, bboxes] = yolov8SegPredict(image,numClasses)

% Copyright 2025 The MathWorks, Inc.

% Load pretrained network.
persistent net
if isempty(net)
    net = coder.loadDeepLearningNetwork('yolov8mSeg.mat');
end

% Get the input size of the network.
inputSize = [640 640 3];

% Apply Preprocessing on the input image.
origSize = size(image);
Ibgr = image(:,:,[3,2,1]);
img = helper.preprocess(Ibgr, inputSize);
newSize = size(img);
img = img(:,:,[3,2,1]);

% Convert to dlarray.
dlInput = dlarray(img, 'SSCB');

% Perform prediction on the input image.
outFeatureMaps = cell(7,1);
[outFeatureMaps{:}] = predict(net, dlInput);

% Apply postprocessing on the output feature maps.
[masks,labelIds, scores, bboxes] = helper.postprocessYOLOv8Seg(outFeatureMaps, ...
    origSize, newSize, numClasses);
end