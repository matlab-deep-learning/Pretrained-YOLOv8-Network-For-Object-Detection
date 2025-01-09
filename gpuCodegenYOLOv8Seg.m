% Read test image
I = imread(fullfile('data','inputTeam.jpg'));

% Get classnames of COCO dataset
classNames = helper.getCOCOClassNames;
numClasses = size(classNames,1);

% Replace 'yolov8n' with other supported models in 'yolov8SegPredict.m' to
% generate code for other YOLO v8 variants
modelName = 'yolov8m';

% Display yolov8SegPredict function
type('yolov8SegPredict.m');

% Generate CUDA Mex
cfg = coder.gpuConfig('mex');
cfg.TargetLang = 'C++';
cfg.DeepLearningConfig = coder.DeepLearningConfig('cudnn');
inputArgs = {I,coder.Constant(numClasses)};
codegen -config cfg yolov8SegPredict -args inputArgs -report

% Perform detection using pretrained model
[masks,labelIds, scores, bboxes] = yolov8SegPredict_mex(I,numClasses);

% Map labelIds back to labels
labels = classNames(labelIds);

Idisp = insertObjectAnnotation(I,"rectangle",bboxes,labels);
numMasks = size(masks,3);
overlayedImage = insertObjectMask(Idisp,masks,MaskColor=lines(numMasks));
figure;imshow(overlayedImage);