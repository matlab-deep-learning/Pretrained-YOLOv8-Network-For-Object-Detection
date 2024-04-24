classdef(SharedTestFixtures = {DownloadYolov8Fixture}) tdownloadPretrainedYOLOv8 < matlab.unittest.TestCase
% Test for downloadPretrainedYOLOv8
    
    % Copyright 2024 The MathWorks, Inc.
    
    
    % The shared test fixture DownloadYolov8Fixture calls
    % downloadPretrainedYOLOv8. Here we check that the downloaded files
    % exists in the appropriate location.
    
    properties        
        DataDir = fullfile(getRepoRoot(),'models');
    end
    
     
    properties(TestParameter)
        Model = {'yolov8n','yolov8s','yolov8m','yolov8l','yolov8x'};
    end
    
    methods(Test)
        function verifyDownloadedFilesExist(test,Model)
            dataFileName = [Model,'.mat'];
            test.verifyTrue(isequal(exist(fullfile(test.DataDir,dataFileName),'file'),2));
        end
    end
end