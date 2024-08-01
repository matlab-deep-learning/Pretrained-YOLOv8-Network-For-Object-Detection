function [bboxes,scores,labelIds] = extractDetections(predictions,numClasses,confThreshold)

% Filter predictions by confidence threshold 
xc = max(predictions(5:end,:),[],1) > confThreshold; % candidates for each image

% TODO: Hardcoding now only one image 
% INDEXING REQUIRED if multiple images 
pred = predictions';
pred = pred(xc(1,:),:);

% box, class
box = pred(:,1:4);
cls = pred(:,5:4+numClasses);

% convert [x_c y_c w h] to [xTopLeft yTopLeft w h]
bboxesTopLeft = iConvertCenterToTopLeft(box); 
[classProbs, classIdx] = max(cls,[],2);
scorePred = classProbs;
classPred = classIdx;

% NMS 
[bboxes, scores, labelIds] = selectStrongestBboxMulticlass(bboxesTopLeft, scorePred, classPred ,...
     'RatioType', 'Union', 'OverlapThreshold', 0.45);
end


function bboxes = iConvertCenterToTopLeft(bboxes)
bboxes(:,1) = bboxes(:,1)- bboxes(:,3)/2 + 0.5; 
bboxes(:,2) = bboxes(:,2)- bboxes(:,4)/2 + 0.5;
bboxes = floor(bboxes);
bboxes(bboxes<1)=1; 
end