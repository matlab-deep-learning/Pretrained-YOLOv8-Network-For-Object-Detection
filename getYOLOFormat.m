% Download data in MATLAB format
downloadObjectData()

%%
outputFolder = fullfile(tempdir,"indoorObjectDetection"); 

% Create image datastore.
datapath = fullfile(outputFolder,"Indoor Object Detection Dataset");
imds = imageDatastore(datapath,IncludeSubfolders=true, FileExtensions=".jpg");

if ~exist('annotationsIndoor.mat','file')
    error('Annotations file is missing. Copy the file from, Multiclass Object Detection Using YOLO v2 Deep Learning example')
    % openExample('deeplearning_shared/MulticlassObjectDetectionUsingDeepLearningExample')
end

% Create box label datastore.
data = load("annotationsIndoor.mat");
blds = data.BBstore;
trainingIdx = data.trainingIdx;
validationIdx = data.validationIdx;
testIdx = data.testIdx;
cleanIdx = data.idxs;

% Remove the 6 images with no labels.
imds = subset(imds,cleanIdx);
blds = subset(blds,cleanIdx);

%%
idx = 1;
while hasdata(imds)
    [I, info] = read(imds);
    strData = split(info.Filename,'/');

    % Works only when extension will be 4 chars, e.g: .jpg
    fileName = strData{end,1}(1:end-4);

    if (idx>100 && idx<150)
        % Use some images as validation data.
        yolov8TrainImagePath = 'datasets/images/val';
        if ~exist(yolov8TrainImagePath,'dir')
            mkdir(yolov8TrainImagePath);
        end
        yolov8TrainLabelPath = 'datasets/labels/val';
        if ~exist(yolov8TrainLabelPath,'dir')
            mkdir(yolov8TrainLabelPath);
        end
    else
        % Other images will be used as training data.
        yolov8TrainImagePath = 'datasets/images/train';
        if ~exist(yolov8TrainImagePath,'dir')
            mkdir(yolov8TrainImagePath);
        end        
        yolov8TrainLabelPath = 'datasets/labels/train';
        if ~exist(yolov8TrainLabelPath,'dir')
            mkdir(yolov8TrainLabelPath);
        end              
    end

    % Save image.
    imwrite(I,[yolov8TrainImagePath,'/',strData{end,1}],"jpg");

    fileID = fopen([yolov8TrainLabelPath,'/',fileName,'.txt'],'w');

    % Read ground truth data.
    gtData = read(blds);

    for i = 1:size(gtData{1,2},1)
        gtLabel = string(gtData{1, 2}(i,1));

        % Following labels should be given in data.yaml in the same order.
        switch gtLabel
            case "exit"
                gtIdx = 0;
            case "fireextinguisher"
                gtIdx = 1;
            case "chair"
                gtIdx = 2;
            case "clock"
                gtIdx = 3;
            case "trashbin"
                gtIdx = 4;
            case "screen"
                gtIdx = 5;
            case "printer"
                gtIdx = 6;
            otherwise
                disp('other value')
        end
        gt = gtData{1,1}(i,:);

        % Normalize groundTruth data.
        gtNormalized = normalizedGTBoxes(I,gt);

        fprintf(fileID,'%d %f %f %f %f\n',[gtIdx gtNormalized]);
    end

    fclose(fileID);
    clc;idx
    idx = idx+1;
end

function scaledGT = normalizedGTBoxes(inpImage, gt)
% The normalizedGTBoxes function applies box transformation on the ground
% truth data and returns normalized bounding boxes.

[Irow, Icol, ~] = size(inpImage);
scaledCords = X1Y1WHToxywh(gt);

scaledGT = scaleGT(scaledCords,Irow,Icol);
end


function normGT = scaleGT(scaledCords, Irow, Icol)
normGT(:,1) = scaledCords(:,1)./Icol;
normGT(:,3) = scaledCords(:,3)./Icol;

normGT(:,2) = scaledCords(:,2)./Irow;
normGT(:,4) = scaledCords(:,4)./Irow;
end

function boxVertices = X1Y1WHToxywh(boxes)
% Convert [x y w h] to [xCenter yCenter w h]. Input and output boxes
% are in pixel coordinates. boxes is an M-by-4 matrix.
boxVertices = boxes;
boxVertices(:,1) = boxes(:,1) + (boxes(:,3)./2);
boxVertices(:,2) = boxes(:,2) + (boxes(:,4)./2);
end

function downloadObjectData()
dsURL = "https://zenodo.org/record/2654485/files/Indoor%20Object%20Detection%20Dataset.zip?download=1"; 
outputFolder = fullfile(tempdir,"indoorObjectDetection"); 
imagesZip = fullfile(outputFolder,"indoor.zip");

if ~exist(imagesZip,"file")   
    mkdir(outputFolder)       
    disp("Downloading 401 MB Indoor Objects Dataset images...") 
    websave(imagesZip,dsURL)
    unzip(imagesZip,fullfile(outputFolder))  
end
end

