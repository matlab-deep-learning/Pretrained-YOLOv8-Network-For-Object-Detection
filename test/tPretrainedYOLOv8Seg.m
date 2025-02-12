classdef(SharedTestFixtures = {DownloadYolov8SegFixture}) tPretrainedYOLOv8Seg < matlab.unittest.TestCase
    % (SharedTestFixtures = {DownloadYolov8Fixture})
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
            detector = load(fullfile(test.RepoRoot,Model.dataFileName));
            image = imread("inputTeam.jpg");

            [masks,labels, scores, bboxes] = segmentObjects(yolov8(detector.net),image);

            test.verifyEqual(size(masks),Model.expectedMasks);
            test.verifyEqual(bboxes, Model.expectedBboxes,'AbsTol',single(1e-4));
            test.verifyEqual(scores, Model.expectedScores,'AbsTol',single(1e-4));
            test.verifyEqual(labels, Model.expectedLabels);
        end
    end
end

function Model = iGetDifferentModels()
% Load Yolov8-n
dataFileName = 'yolov8nSeg.mat';

% Expected detection results.
expectedMasks = [413 503 4];
expectedBboxes = single([154.9498   30.2393  109.5871  370.6212;...
  264.4729   32.2021  119.3134  355.7670;...
  388.5913   50.3516  111.2383  318.7885;...
   29.6955   52.0111  140.3341  357.4400]);
expectedScores = single([0.9057; 0.8791; 0.9197; 0.8945]);
expectedLabels = categorical({'person';'person';'person';'person'});

detectorYolov8n = struct('dataFileName',dataFileName,...
    'expectedMasks',expectedMasks,'expectedBboxes',expectedBboxes, ...
    'expectedScores',expectedScores,'expectedLabels',expectedLabels);

% Load Yolov8-s
dataFileName = 'yolov8sSeg.mat';

% Expected detection results.
expectedMasks = [413 503 4];
expectedBboxes = single([154.6472   31.5058  109.4336  371.4240;...
   30.9926   53.2941  137.9434  355.7844;...
  262.7715   34.1726  120.8975  353.6561;...
  389.0290   50.3817  109.8241  321.9294]);
expectedScores = single([0.9299; 0.9007; 0.9167; 0.9388]);
expectedLabels = categorical({'person';'person';'person';'person'});

detectorYolov8s = struct('dataFileName',dataFileName,...
    'expectedMasks',expectedMasks,'expectedBboxes',expectedBboxes, ...
    'expectedScores',expectedScores,'expectedLabels',expectedLabels);

% Load Yolov8-m
dataFileName = 'yolov8mSeg.mat';

% Expected detection results.
expectedMasks = [413 503 4];
expectedBboxes = single([154.9606   30.5550  109.5372  372.2964;...
  389.1360   50.4475  110.6815  323.4165;...
  263.8429   32.4916  120.3857  357.2574;...
   29.5446   52.2939  140.7661  357.2836]);
expectedScores = single([0.9415; 0.9516; 0.9491; 0.9255]);
expectedLabels = categorical({'person';'person';'person';'person'});

detectorYolov8m = struct('dataFileName',dataFileName,...
    'expectedMasks',expectedMasks,'expectedBboxes',expectedBboxes, ...
    'expectedScores',expectedScores,'expectedLabels',expectedLabels);

% Load Yolov8-l
dataFileName = 'yolov8lSeg.mat';

% Expected detection results.
expectedMasks = [413 503 4];
expectedBboxes = single([389.1389   50.6288  109.9677  324.0468;...
   28.0494   53.2758  141.0950  356.1628;...
  155.2541   30.9948  109.6235  373.9222;...
  262.9318   33.9963  120.8235  353.9915]);
expectedScores = single([0.9373; 0.9378; 0.9366; 0.9412]);
expectedLabels = categorical({'person';'person';'person';'person'});

detectorYolov8l = struct('dataFileName',dataFileName,...
    'expectedMasks',expectedMasks,'expectedBboxes',expectedBboxes, ...
    'expectedScores',expectedScores,'expectedLabels',expectedLabels);

% Load Yolov8-x
dataFileName = 'yolov8xSeg.mat';

% Expected detection results.
expectedMasks = [413 503 4];
expectedBboxes = single([262.6409   34.3788  121.3496  352.7583;...
  155.2030   30.5582  109.3741  373.6773;...
  388.8879   49.2623  109.6951  326.3129;...
   29.3609   53.0440  141.1476  355.5334]);
expectedScores = single([0.9419; 0.9257; 0.9511; 0.9369]);
expectedLabels = categorical({'person';'person';'person';'person'});

detectorYolov8x = struct('dataFileName',dataFileName,...
    'expectedMasks',expectedMasks,'expectedBboxes',expectedBboxes, ...
    'expectedScores',expectedScores,'expectedLabels',expectedLabels);

Model = struct(...
    'detectorYolov8n',detectorYolov8n,'detectorYolov8s',detectorYolov8s,'detectorYolov8m',...
    detectorYolov8m,'detectorYolov8l',detectorYolov8l,'detectorYolov8x',detectorYolov8x);

end