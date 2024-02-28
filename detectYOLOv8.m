function [bboxes, scores, labelIds] = detectYOLOv8(dlnet, image, numClasses, executionEnvironment)
% detectYOLOv8 runs prediction on a trained yolov8 network.
%
% Inputs:
% dlnet                - Pretrained yolov8 dlnetwork.
% image                - RGB image to run prediction on. (H x W x 3)
% numClasses           - Number of classes to be detected.
% executionEnvironment - Environment to run predictions on. Specify cpu,
%                        gpu, or auto.
%
% Outputs:
% bboxes     - Final bounding box detections ([x y w h]) formatted as
%              NumDetections x 4.
% scores     - NumDetections x 1 classification scores.
% labelIds   - NumDetections x 1 label Ids.

% Copyright 2024 The MathWorks, Inc

% Get the input size of the network.
inputSize = dlnet.Layers(1).InputSize;

% Apply Preprocessing on the input image.
origSize = size(image);
Ibgr = image(:,:,[3,2,1]); % convert image to bgr
img = helper.preprocess(Ibgr, inputSize);

newSize = size(img);
img = img(:,:,[3,2,1]); % convert image to rgb

% Convert to dlarray.
dlInput = dlarray(img, 'SSCB');

% If GPU is available, then convert data to gpuArray.
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlInput = gpuArray(dlInput);
end

% Perform prediction on the input image.
outFeatureMaps = cell(length(dlnet.OutputNames), 1);
[outFeatureMaps{:}] = predict(dlnet, dlInput);

% Apply postprocessing on the output feature maps.
[bboxes,scores,labelIds] = helper.postprocess(outFeatureMaps, ...
     origSize, newSize, numClasses);

bboxes = gather(bboxes);
scores = gather(scores);
labelIds = gather(labelIds);

end
