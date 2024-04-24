classdef(SharedTestFixtures = {DownloadYolov8Fixture}) tload < matlab.unittest.TestCase
    % Test to load the the downloaded models.

    % Copyright 2024 The MathWorks, Inc.

    % The shared test fixture DownloadYolov4Fixture calls
    % downloadPretrainedYOLOv4. Here we check the properties of
    % downloaded models.

    properties
        DataDir = fullfile(getRepoRoot(),'models');
    end

    properties(TestParameter)
        Model = iGetDifferentModels();
    end

    methods(Test)
        function verifyModelAndFields(test,Model)
            % Test point to verify the fields of the downloaded models are
            % as expected.
            loadedModel = load(fullfile(test.DataDir,Model.dataFileName));

            test.verifyClass(loadedModel.yolov8Net,'dlnetwork');
            test.verifyEqual(numel(loadedModel.yolov8Net.Layers), Model.expectedNumLayers);
            test.verifyEqual(size(loadedModel.yolov8Net.Connections), Model.expectedConnectionsSize);
            test.verifyEqual(loadedModel.yolov8Net.InputNames, Model.expectedInputNames);
            test.verifyEqual(loadedModel.yolov8Net.OutputNames, Model.expectedOutputNames);
        end
    end
end

function Model = iGetDifferentModels()
    % Load Yolov8-n
    dataFileName = 'yolov8n.mat';

    expectedNumLayers = 213;
    expectedConnectionsSize = [303, 2];
    expectedInputNames = {{'images'}};
    expectedOutputNames = {{'x_model_22_Concat' 'x_model_22_Concat_1' 'x_model_22_Concat_2'}};
    detectorYolov8n = struct('dataFileName',dataFileName,...
    'expectedNumLayers',expectedNumLayers,'expectedConnectionsSize',expectedConnectionsSize,...
    'expectedInputNames',expectedInputNames, 'expectedOutputNames',expectedOutputNames);

    % Load Yolov8-s
    dataFileName = 'yolov8s.mat';

    expectedNumLayers = 213;
    expectedConnectionsSize = [303, 2];
    expectedInputNames = {{'images'}};
    expectedOutputNames = {{'x_model_22_Concat' 'x_model_22_Concat_1' 'x_model_22_Concat_2'}};
    detectorYolov8s = struct('dataFileName',dataFileName,...
    'expectedNumLayers',expectedNumLayers,'expectedConnectionsSize',expectedConnectionsSize,...
    'expectedInputNames',expectedInputNames, 'expectedOutputNames',expectedOutputNames);

    % Load Yolov8-m
    dataFileName = 'yolov8m.mat';

    expectedNumLayers = 279;
    expectedConnectionsSize = [405, 2];
    expectedInputNames = {{'images'}};
    expectedOutputNames = {{'x_model_22_Concat' 'x_model_22_Concat_1' 'x_model_22_Concat_2'}};
    detectorYolov8m = struct('dataFileName',dataFileName,...
    'expectedNumLayers',expectedNumLayers,'expectedConnectionsSize',expectedConnectionsSize,...
    'expectedInputNames',expectedInputNames, 'expectedOutputNames',expectedOutputNames);

    % Load Yolov8-l
    dataFileName = 'yolov8l.mat';

    expectedNumLayers = 345;
    expectedConnectionsSize = [507, 2];
    expectedInputNames = {{'images'}};
    expectedOutputNames = {{'x_model_22_Concat' 'x_model_22_Concat_1' 'x_model_22_Concat_2'}};
    detectorYolov8l = struct('dataFileName',dataFileName,...
    'expectedNumLayers',expectedNumLayers,'expectedConnectionsSize',expectedConnectionsSize,...
    'expectedInputNames',expectedInputNames, 'expectedOutputNames',expectedOutputNames);

    % Load Yolov8-x
    dataFileName = 'yolov8x.mat';

    expectedNumLayers = 346;
    expectedConnectionsSize = [508, 2];
    expectedInputNames = {{'images'}};
    expectedOutputNames = {{'x_model_22_Concat' 'x_model_22_Concat_1' 'x_model_22_Concat_2'}};
    detectorYolov8x = struct('dataFileName',dataFileName,...
    'expectedNumLayers',expectedNumLayers,'expectedConnectionsSize',expectedConnectionsSize,...
    'expectedInputNames',expectedInputNames, 'expectedOutputNames',expectedOutputNames);

    Model = struct(...
        'detectorYolov8n',detectorYolov8n,'detectorYolov8s',detectorYolov8s,'detectorYolov8m',...
        detectorYolov8m,'detectorYolov8l',detectorYolov8l,'detectorYolov8x',detectorYolov8x);

end