classdef DownloadYolov8Fixture < matlab.unittest.fixtures.Fixture
    % DownloadYolov8Fixture   A fixture for calling downloadPretrainedYOLOv8
    % if necessary. This is to ensure that this function is only called once
    % and only when tests need it. It also provides a teardown to return
    % the test environment to the expected state before testing.

    % Copyright 2024 The MathWorks, Inc

    properties(Constant)
        Yolov8DataDir = fullfile(getRepoRoot(),'models')
    end

    properties        
        yolov8sExist (1,1)logical
        yolov8mExist (1,1)logical
        yolov8lExist (1,1)logical
        yolov8xExist (1,1)logical
    end
    methods
        function setup(this)            
            this.yolov8sExist = exist(fullfile(this.Yolov8DataDir,'yolov8s.mat'),'file')==2;
            this.yolov8mExist = exist(fullfile(this.Yolov8DataDir,'yolov8m.mat'),'file')==2;
            this.yolov8lExist = exist(fullfile(this.Yolov8DataDir,'yolov8l.mat'),'file')==2;
            this.yolov8xExist = exist(fullfile(this.Yolov8DataDir,'yolov8x.mat'),'file')==2;

            % Call this in eval to capture and drop any standard output
            % that we don't want polluting the test logs.

            % The downloadPretrained function doesnot exist. Uncomment and
            % add the names ones the file in added.            
            if ~this.yolov8sExist
            	evalc('helper.downloadPretrainedYOLOv8(''yolov8s'');');                
            end
            if ~this.yolov8mExist
            	evalc('helper.downloadPretrainedYOLOv8(''yolov8m'');');                
            end
            if ~this.yolov8lExist
            	evalc('helper.downloadPretrainedYOLOv8(''yolov8l'');');                
            end
            if ~this.yolov8xExist
            	evalc('helper.downloadPretrainedYOLOv8(''yolov8x'');');                
            end

        end
        function teardown(this)
            if ~this.yolov8sExist
            	delete(fullfile(this.Yolov8DataDir,'yolov8s.mat'));               
            end
            if ~this.yolov8mExist
            	delete(fullfile(this.Yolov8DataDir,'yolov8m.mat'));                
            end
            if ~this.yolov8lExist
            	delete(fullfile(this.Yolov8DataDir,'yolov8l.mat'));               
            end
            if ~this.yolov8xExist
            	delete(fullfile(this.Yolov8DataDir,'yolov8x.mat'));                
            end
        end
    end
end
