classdef DownloadYolov8SegFixture < matlab.unittest.fixtures.Fixture
    % DownloadYolov8Fixture   A fixture for calling downloadPretrainedYOLOv8
    % if necessary. This is to ensure that this function is only called once
    % and only when tests need it. It also provides a teardown to return
    % the test environment to the expected state before testing.

    % Copyright 2024 The MathWorks, Inc

    properties(Constant)
        Yolov8DataDir = fullfile(getRepoRoot())
    end

    properties        
        yolov8nSegExist (1,1)logical
        yolov8sSegExist (1,1)logical
        yolov8mSegExist (1,1)logical
        yolov8lSegExist (1,1)logical
        yolov8xSegExist (1,1)logical
    end
    methods
        function setup(this)            
            this.yolov8nSegExist = exist(fullfile(this.Yolov8DataDir,'yolov8nSeg.mat'),'file')==2;
            this.yolov8sSegExist = exist(fullfile(this.Yolov8DataDir,'yolov8sSeg.mat'),'file')==2;
            this.yolov8mSegExist = exist(fullfile(this.Yolov8DataDir,'yolov8mSeg.mat'),'file')==2;
            this.yolov8lSegExist = exist(fullfile(this.Yolov8DataDir,'yolov8lSeg.mat'),'file')==2;
            this.yolov8xSegExist = exist(fullfile(this.Yolov8DataDir,'yolov8xSeg.mat'),'file')==2;

            % Call this in eval to capture and drop any standard output
            % that we don't want polluting the test logs.

            % The downloadPretrained function doesnot exist. Uncomment and
            % add the names ones the file in added. 
            if ~this.yolov8nSegExist
            	evalc('yolov8(''yolov8n'');');                
            end
            if ~this.yolov8sSegExist
            	evalc('yolov8(''yolov8s'');');                
            end
            if ~this.yolov8mSegExist
            	evalc('yolov8(''yolov8m'');');                
            end
            if ~this.yolov8lSegExist
            	evalc('yolov8(''yolov8l'');');                
            end
            if ~this.yolov8xSegExist
            	evalc('yolov8(''yolov8x'');');                
            end

        end
        function teardown(this)
            if ~this.yolov8nSegExist
            	delete(fullfile(this.Yolov8DataDir,'yolov8nSeg.mat'));               
            end
            if ~this.yolov8sSegExist
            	delete(fullfile(this.Yolov8DataDir,'yolov8sSeg.mat'));               
            end
            if ~this.yolov8mSegExist
            	delete(fullfile(this.Yolov8DataDir,'yolov8mSeg.mat'));                
            end
            if ~this.yolov8lSegExist
            	delete(fullfile(this.Yolov8DataDir,'yolov8lSeg.mat'));               
            end
            if ~this.yolov8xSegExist
            	delete(fullfile(this.Yolov8DataDir,'yolov8xSeg.mat'));                
            end
        end
    end
end
