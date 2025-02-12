classdef(SharedTestFixtures = {DownloadYolov8SegFixture}) tloadSeg < matlab.unittest.TestCase
    % Test to load the the downloaded models.

    % Copyright 2024 The MathWorks, Inc.

    % The shared test fixture DownloadYolov4Fixture calls
    % downloadPretrainedYOLOv4. Here we check the properties of
    % downloaded models.

    properties
        DataDir = fullfile(getRepoRoot());
    end

    properties(TestParameter)
        Model = iGetDifferentModels();
    end

    methods(Test)
        function verifyModelAndFields(test,Model)
            % Test point to verify the fields of the downloaded models are
            % as expected.
            loadedModel = load(fullfile(test.DataDir,Model.dataFileName));
            test.verifyClass(loadedModel.net,'dlnetwork');
            test.verifyEqual(numel(loadedModel.net.Layers), Model.expectedNumLayers);
            test.verifyEqual(size(loadedModel.net.Connections), Model.expectedConnectionsSize);
            test.verifyEqual(loadedModel.net.InputNames, Model.expectedInputNames);
            test.verifyEqual(loadedModel.net.OutputNames, Model.expectedOutputNames);
        end
    end
end

function Model = iGetDifferentModels()
    % Load Yolov8-n
    dataFileName = 'yolov8nSeg.mat';

    expectedNumLayers = 178;
    expectedConnectionsSize = [211, 2];
    expectedInputNames = {{'images'}};
    expectedOutputNames = {{'x_model_22_cv4_0__13' 'x_model_22_cv4_1__13' 'x_model_22_cv4_2__13' 'x_model_22_Concat_1' ['x_model_22_Conc' ...
        'at_2'] 'x_model_22_Concat_3' 'x_model_22_proto_c_7'}};
    detectorYolov8n = struct('dataFileName',dataFileName,...
    'expectedNumLayers',expectedNumLayers,'expectedConnectionsSize',expectedConnectionsSize,...
    'expectedInputNames',expectedInputNames, 'expectedOutputNames',expectedOutputNames);

    % Load Yolov8-s
    dataFileName = 'yolov8sSeg.mat';

    expectedNumLayers = 178;
    expectedConnectionsSize = [211, 2];
    expectedInputNames = {{'images'}};
    expectedOutputNames = {{'x_model_22_cv4_0__13' 'x_model_22_cv4_1__13' 'x_model_22_cv4_2__13' 'x_model_22_Concat_1' ['x_model_22_Conc' ...
        'at_2'] 'x_model_22_Concat_3' 'x_model_22_proto_c_7'}};
    detectorYolov8s = struct('dataFileName',dataFileName,...
    'expectedNumLayers',expectedNumLayers,'expectedConnectionsSize',expectedConnectionsSize,...
    'expectedInputNames',expectedInputNames, 'expectedOutputNames',expectedOutputNames);

    % Load Yolov8-m
    dataFileName = 'yolov8mSeg.mat';

    expectedNumLayers = 224;
    expectedConnectionsSize = [273, 2];
    expectedInputNames = {{'images'}};
    expectedOutputNames = {{'x_model_22_cv4_0__13' 'x_model_22_cv4_1__13' 'x_model_22_cv4_2__13' 'x_model_22_Concat_1' ['x_model_22_Conc' ...
        'at_2'] 'x_model_22_Concat_3' 'x_model_22_proto_c_7'}};
    detectorYolov8m = struct('dataFileName',dataFileName,...
    'expectedNumLayers',expectedNumLayers,'expectedConnectionsSize',expectedConnectionsSize,...
    'expectedInputNames',expectedInputNames, 'expectedOutputNames',expectedOutputNames);

    % Load Yolov8-l
    dataFileName = 'yolov8lSeg.mat';

    expectedNumLayers = 270;
    expectedConnectionsSize = [335, 2];
    expectedInputNames = {{'images'}};
    expectedOutputNames = {{'x_model_22_cv4_0__13' 'x_model_22_cv4_1__13' 'x_model_22_cv4_2__13' 'x_model_22_Concat_1' ['x_model_22_Conc' ...
        'at_2'] 'x_model_22_Concat_3' 'x_model_22_proto_c_7'}};
    detectorYolov8l = struct('dataFileName',dataFileName,...
    'expectedNumLayers',expectedNumLayers,'expectedConnectionsSize',expectedConnectionsSize,...
    'expectedInputNames',expectedInputNames, 'expectedOutputNames',expectedOutputNames);

    % Load Yolov8-x
    dataFileName = 'yolov8xSeg.mat';

    expectedNumLayers = 270;
    expectedConnectionsSize = [335, 2];
    expectedInputNames = {{'images'}};
    expectedOutputNames = {{'x_model_22_cv4_0__13' 'x_model_22_cv4_1__13' 'x_model_22_cv4_2__13' 'x_model_22_Concat_1' ['x_model_22_Conc' ...
        'at_2'] 'x_model_22_Concat_3' 'x_model_22_proto_c_7'}};
    detectorYolov8x = struct('dataFileName',dataFileName,...
    'expectedNumLayers',expectedNumLayers,'expectedConnectionsSize',expectedConnectionsSize,...
    'expectedInputNames',expectedInputNames, 'expectedOutputNames',expectedOutputNames);

    Model = struct(...
        'detectorYolov8n',detectorYolov8n,'detectorYolov8s',detectorYolov8s,'detectorYolov8m',...
        detectorYolov8m,'detectorYolov8l',detectorYolov8l,'detectorYolov8x',detectorYolov8x);

end