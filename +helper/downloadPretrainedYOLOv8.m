function model = downloadPretrainedYOLOv8(modelName)
% The downloadPretrainedYOLOv8 function downloads a YOLO v8 network 
% pretrained on COCO dataset.
%
% Copyright 2024 The MathWorks, Inc.

supportedNetworks = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"];
validatestring(modelName, supportedNetworks);

dataPath = 'models';
netMatFileFullPath = fullfile(dataPath, [modelName, '.mat']);

if ~exist(netMatFileFullPath,'file')
    fprintf(['Downloading pretrained ', modelName ,' network.\n']);
    fprintf('This can take several minutes to download...\n');
    url = ['https://github.com/matlab-deep-learning/Pretrained-YOLOv8-Network-For-Object-Detection/releases/download/v1.0.0/', modelName, '.mat'];
    websave(netMatFileFullPath, url);
    fprintf('Done.\n\n');
else
    fprintf(['Pretrained ', modelName, ' network already exists.\n\n']);
end

model = load(netMatFileFullPath);
end
