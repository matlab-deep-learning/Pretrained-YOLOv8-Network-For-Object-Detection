% Add path containing the pretrained models.
addpath('models');

% Read test image.
I = imread(fullfile('data','inputTeam.jpg'));

% Get classnames of COCO dataset.
classNames = helper.getCOCOClassNames;
numClasses = size(classNames,1);

% Replace 'yolov8n' with other supported models in yolov8Predict to generate code for
% other YOLO v8 variants.
modelName = 'yolov8n';

% Display yolov8Predict function.
type('yolov8Predict.m');

% Generate CUDA Mex.
cfg = coder.gpuConfig('mex');
cfg.TargetLang = 'C++';
cfg.DeepLearningConfig = coder.DeepLearningConfig('cudnn');
inputArgs = {I,coder.Constant(numClasses)};
codegen -config cfg yolov8Predict -args inputArgs -report

% Perform detection using pretrained model.
[bboxes,scores,labelIds] = yolov8Predict_mex(I,numClasses);

% Map labelIds back to labels.
labels = classNames(labelIds);

% Visualize detection results.
annotations = string(labels) + ": " + string(scores);
Iout = insertObjectAnnotation(I, 'rectangle', bboxes, annotations);
figure 
imshow(Iout);
