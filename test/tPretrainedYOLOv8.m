classdef(SharedTestFixtures = {DownloadYolov8Fixture}) tPretrainedYOLOv8 < matlab.unittest.TestCase
    % Test to detect accuracy of the downloaded models.

    % Copyright 2024 The MathWorks, Inc.

    % The shared test fixture DownloadYolov4Fixture calls
    % downloadPretrainedYOLOv4. Here we check the properties of
    % downloaded models.

    properties
        RepoRoot = getRepoRoot;
    end

    properties(TestParameter)
        Model = iGetDifferentModels();
    end

    methods(Test)
        function verifyModelAndFields(test,Model)
            % Test point to verify the fields of the downloaded models are
            % as expected.
            detector = load(fullfile(test.RepoRoot,'models',Model.dataFileName));
            image = imread("inputTeam.jpg");
            classNames = helper.getCOCOClassNames;
            numClasses = size(classNames,1);
            [bboxes, scores, labelIDs] = detectYOLOv8(detector.yolov8Net,image,numClasses,"auto");
            labels = classNames(labelIDs);

            test.verifyEqual(bboxes, Model.expectedBboxes,'AbsTol',single(1e-4));
            test.verifyEqual(scores, Model.expectedScores,'AbsTol',single(1e-4));
            test.verifyEqual(labels, Model.expectedLabels);
        end
    end
end

function Model = iGetDifferentModels()
% Load Yolov8-n
dataFileName = 'yolov8n.mat';

% Expected detection results.
expectedBboxes = single([154.8297   29.6641  108.6734  371.1765;...
  389.8250   50.0984  107.1016  319.3047;...
   30.6516   50.0984  133.8234  358.6015;...
  261.7172   32.8078  121.2484  353.8860]);
expectedScores = single([0.9152; 0.9295; 0.8889; 0.8968]);
expectedLabels = categorical({'person';'person';'person';'person'});

detectorYolov8n = struct('dataFileName',dataFileName,...
    'expectedBboxes',expectedBboxes,'expectedScores',expectedScores,...
    'expectedLabels',expectedLabels);

% Load Yolov8-s
dataFileName = 'yolov8s.mat';

% Expected detection results.
expectedBboxes = single([263.2891   33.5938  119.6766  352.3141;...
  389.0391   50.0984  109.4594  318.5187;...
  154.8297   32.0219  108.6734  372.7484;...
   29.8656   50.8844  140.1109  357.0297]);
expectedScores = single([0.9054; 0.9439; 0.9114; 0.9062]);
expectedLabels = categorical({'person';'person';'person';'person'});

detectorYolov8s = struct('dataFileName',dataFileName,...
    'expectedBboxes',expectedBboxes,'expectedScores',expectedScores,...
    'expectedLabels',expectedLabels);

% Load Yolov8-m
dataFileName = 'yolov8m.mat';

% Expected detection results.
expectedBboxes = single([155.6156   30.4500  107.8875  372.7484;...
  263.2891   32.8078  119.6766  356.2437;...
  389.0391   50.8844  108.6734  320.8766;...
   29.0797   52.4562  140.8969  357.0297]);
expectedScores = single([0.9325; 0.9349; 0.9483; 0.9308]);
expectedLabels = categorical({'person';'person';'person';'person'});

detectorYolov8m = struct('dataFileName',dataFileName,...
    'expectedBboxes',expectedBboxes,'expectedScores',expectedScores,...
    'expectedLabels',expectedLabels);

% Load Yolov8-l
dataFileName = 'yolov8l.mat';

% Expected detection results.
expectedBboxes = single([154.8297   30.4500  108.6734  373.5344;...
  389.0391   50.0984  108.6734  321.6625;...
  263.2891   33.5938  119.6766  354.6719;...
   29.0797   52.4562  140.8969  354.6719]);
expectedScores = single([0.9394; 0.9370; 0.9390; 0.9281]);
expectedLabels = categorical({'person';'person';'person';'person'});

detectorYolov8l = struct('dataFileName',dataFileName,...
    'expectedBboxes',expectedBboxes,'expectedScores',expectedScores,...
    'expectedLabels',expectedLabels);

% Load Yolov8-x
dataFileName = 'yolov8x.mat';

% Expected detection results.
expectedBboxes = single([154.8297   30.4500  108.6734  372.7484;...
  389.0391   50.0984  108.6734  323.2343;...
  263.2891   35.1656  119.6766  353.1000;...
   28.2938   51.6703  141.6828  354.6719]);
expectedScores = single([0.9421; 0.9425; 0.9469; 0.9355]);
expectedLabels = categorical({'person';'person';'person';'person'});

detectorYolov8x = struct('dataFileName',dataFileName,...
    'expectedBboxes',expectedBboxes,'expectedScores',expectedScores,...
    'expectedLabels',expectedLabels);

Model = struct(...
    'detectorYolov8n',detectorYolov8n,'detectorYolov8s',detectorYolov8s,'detectorYolov8m',...
    detectorYolov8m,'detectorYolov8l',detectorYolov8l,'detectorYolov8x',detectorYolov8x);

end