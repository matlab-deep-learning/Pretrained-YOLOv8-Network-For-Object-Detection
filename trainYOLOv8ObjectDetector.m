function yolov8Det = trainYOLOv8ObjectDetector(configFile,baseModel,options)
%trainYOLOv8ObjectDetector Train a YOLO v8 deep learning object detector.
%
% trainedDetector = trainYOLOv8ObjectDetector(configFile, baseModel) trains a
% YOLOv8 object detector using the specified configuration file. The
% configFile is the path to the configuration file in YAML format. The
% options structure can be used to specify the maximum number of epochs and
% the mini-batch size for training. A YOLO v8 object detector can be
% trained to detect multiple object classes.
%
% % Inputs
% --------
%   configFile      Includes the locations of relative path, train, val,
%                   test data specified as string array along with the
%                   class names information along with indices.
%                   Following is the default yaml format and its contents:
%                       path: ''
%                       train: images/train
%                       val: images/val
%                       test: null
%                       names:
%                           0: exit
%                           1: fireextinguisher
%                           2: chair
%                           3: clock
%                           4: trashbin
%                           5: screen
%                           6: printer
%                   Note: null can be used in absence of either val, test data.
%
% % Additional input arguments
% ----------------------------
% [...] = trainYOLOv8ObjectDetector(..., Name = Value) specifies additional
% name-value pair arguments described below:
%
%    'MaxEpochs'        The maximum number of epochs that will be used for
%                       training.
%                  
%                       Default: 10
%
%   'MiniBatchSize'     The size of the mini-batch used for each training
%                       iteration.
%
%                       Default: 16
%
%   'ImageSize'         Specify the image size used for training the
%                       detector. The input size must be H-by-W or
%                       H-by-W-by-C.
%
%                       Default: [680 680 3]
%
% % Example: Train YOLO v8 network
% --------------------------------
% % Create a yolov8Trainer object
%  yolov8Obj = yolov8ObjectDetector("yolov8m.pt")
%
% % Train YOLO v8 detector
% yolov8Obj = trainYOLOv8ObjectDetector (yolov8Obj, 'data.yaml');
%
% See also trainYOLOv4ObjectDetector, trainYOLOv3ObjectDetector
%
% Copyright 2024 The MathWorks, Inc.

arguments
    configFile;
    baseModel {mustBeMember(baseModel, ["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt"])};
    options.MaxEpochs (1,1) {mustBeNumeric, mustBePositive, mustBeReal, mustBeFinite} = 10;
    options.MiniBatchSize (1,1) {mustBeNumeric, mustBePositive, mustBeReal, mustBeFinite} = 16;
    options.ImageSize {mustBeNumeric, mustBePositive, mustBeReal, mustBeFinite} = [640 640 3];
end

if ~canUseGPU()
error("Training of YOLO v8 object detector requires GPU.")
end

terminate(pyenv)
pyenv(Version="glnxa64/python/bin/python3", ExecutionMode = "OutOfProcess")
if isunix
    py.sys.setdlopenflags(int32(bitor(int64(py.os.RTLD_LAZY),int64(py.os.RTLD_DEEPBIND))));
end

pythonObject = py.trainYOLOv8Wrapper.yolov8TrainerClass(py.str(baseModel),py.int(options.ImageSize(1,1))); 

% Train YOLO v8 using config. file
results = pythonObject.trainYOLOv8(configFile,py.int(options.MaxEpochs));

% Obtain path to results dir that has .pt file
getSaveDir = string(results.save_dir.parts);

% Extract location of export directory
onnxExportDir = [getSaveDir,'weights','best.pt'];
onnxExportPath = fullfile(onnxExportDir{:});

% Export best trained model to onnx format
pythonObject.exportModel(onnxExportPath);

% Import the exported model in MATLAB
modelPath = fullfile(onnxExportDir{1:end-1});
net = importYOLOv8Model(fullfile(modelPath,"best.onnx"));

dictValues = pythonObject.getClassNames(configFile);
classDict = dictValues{"names"};
matlabDict = dictionary(classDict);
classNames = values(matlabDict);

yolov8Det = yolov8ObjectDetector(net, classNames);

end
