function [bboxes,scores,labelIds] = postprocess(outFeatureMaps, origSize, newSize, numClasses)
% The postprocess function applies postprocessing on the generated feature
% maps and returns bounding boxes, detection scores and labels.

% Copyright 2024 The MathWorks, Inc.

% Transform outFeatureMaps to box [x_center y_center w h] and class scores
[box cls] = helper.yolov8Transform(outFeatureMaps, numClasses);

% Map network predictions to anchor girds.
predictions = cat(1, box, cls);
confThreshold = 0.5;

% Extract detections from predictions.
[bboxesPred,scores,labelIds] = helper.extractDetections(predictions, numClasses, confThreshold);
bboxesTmp = xywhToX1Y1X2Y2(bboxesPred);

% Map predictions back to original dimension.
bboxesPost = helper.postProcessYOLOv8(newSize, bboxesTmp, origSize);
bboxes = x1y1x2y2ToXYWH(bboxesPost);

end

function boxes = xywhToX1Y1X2Y2(boxes)
% Convert [x y w h] box to [x1 y1 x2 y2]. Input and output
% boxes are in pixel coordinates. boxes is an M-by-4
% matrix.
boxes(:,3) = boxes(:,1) + boxes(:,3) - 1;
boxes(:,4) = boxes(:,2) + boxes(:,4) - 1;
end

function boxes = x1y1x2y2ToXYWH(boxes)
% Convert [x1 y1 x2 y2] boxes into [x y w h] format. Input and
% output boxes are in pixel coordinates. boxes is an M-by-4
% matrix.
boxes(:,3) = boxes(:,3) - boxes(:,1) + 1;
boxes(:,4) = boxes(:,4) - boxes(:,2) + 1;
end
