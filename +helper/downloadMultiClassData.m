function downloadMultiClassData()
% Download data to train YOLO v8 object detector.
if ~exist('datasets.zip', 'file')
    disp('Downloading multiclass object detection dataset...');
    datasetURL = 'https://github.com/matlab-deep-learning/Pretrained-YOLOv8-Network-For-Object-Detection/releases/download/v1.0.0/datasets.zip';
    websave('datasets.zip', datasetURL);
    unzip('datasets.zip');
end
end