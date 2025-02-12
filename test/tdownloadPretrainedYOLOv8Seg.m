classdef(SharedTestFixtures = {DownloadYolov8SegFixture}) tdownloadPretrainedYOLOv8Seg < matlab.unittest.TestCase
% Test for downloadPretrainedYOLOv8
    
    % Copyright 2025 The MathWorks, Inc.
    
    % The shared test fixture DownloadYolov8Fixture calls
    % downloadPretrainedYOLOv8. Here we check that the downloaded files
    % exists in the appropriate location.
    
    properties        
        DataDir = fullfile(getRepoRoot());
    end
    
     
    properties(TestParameter)
        Model = {'yolov8nSeg','yolov8sSeg','yolov8mSeg','yolov8lSeg','yolov8xSeg'};
    end
    
    methods(Test)
        function verifyDownloadedFilesExist(test,Model)
            dataFileName = [Model,'.mat'];
            test.verifyTrue(isequal(exist(fullfile(test.DataDir,dataFileName),'file'),2));
        end
    end
end