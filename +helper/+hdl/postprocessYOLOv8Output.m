function [bboxes,scores,labels] = postprocessYOLOv8Output(hwprediction, img, imgPreprocessed, numClasses)
    initialSize = size(img);
    newSize = size(imgPreprocessed);
    [bboxes,scores,labelIds] = helper.postprocess(hwprediction, ...
     initialSize, newSize, numClasses);

    bboxes = gather(bboxes);
    scores = gather(scores);
    labelIds = gather(labelIds);

    % Map labelIds back to labels.
    labels = classNames(labelIds);
end