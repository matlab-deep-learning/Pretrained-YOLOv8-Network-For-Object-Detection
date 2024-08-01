%yolov8ObjectDetector Detect objects using YOLO v8 deep learning detector.
%
% detector = yolov8ObjectDetector() loads a smaller version of YOLO v8
% object detector trained to detect 80 object classes from the COCO
% dataset.
%
% detector = yolov8ObjectDetector(detectorName) loads a pretrained YOLO v8
% object detector specified by detectorName. detectorName must be either
% 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', or 'yolov8x'.
%
% detector = yolov8ObjectDetector(network, classNames) configures a
% pretrained YOLO v8 object detector for transfer learning with a new set
% of object classes.
%
% Inputs:
% -------
%    detectorName   Specify the name of the pretrained YOLO v8 deep learning
%                   model as a string or character vector. The value must
%                   be one of the following:
%
%                   'yolov8n'   Use this model for speed and efficiency.
%
%                   'yolov8s'   Use this model for a balance between speed
%                               and accuracy, suitable for applications
%                               requiring real-time performance with good
%                               detection quality.
%
%                   'yolov8m'   Use this model for higher accuracy with
%                               moderate computational demands.
%
%                   'yolov8l'   Use this model to prioritize maximum
%                               detection accuracy for high-end systems, at
%                               the cost of computational intensity.
%
%                   'yolov8x'   Use this model to get most accurate
%                               detections but requires significant
%                               computational resources, ideal for high-end
%                               systems prioritizing detection performance.
%
%    classNames     Specify the names of object classes that the YOLO v8
%                   object detector is configured to detect. classNames can
%                   be a string vector, a categorical vector, or a cell
%                   array of character vectors.
%
% % Additional input arguments
% ----------------------------
% [...] = yolov8ObjectDetector(..., Name=Value) specifies additional
% name-value pair arguments described below:
%
%   'ModelName'                      Specify the name for the object detector
%                                    as a string or character vector.
%
%                                    Default: '' or specified detectorName
%
%   'InputSize'                      Specify the image size used for training
%                                    the detector. The input size must be
%                                    H-by-W or H-by-W-by-C.
%
%                                    Default: network input size
%
%   'NormalizationStatistics'       Specifiy z-score normalization
%                                   statitics as a structure with fields,
%                                   Mean and StandardDeviation specified as
%                                   1-by-C array of means and standard
%                                   deviation per channel. The number of
%                                   channels, C must match the InputSize.
%
%                                   Default: 1-by-1 struct with fields
%                                            containing the following values
%                                            for the COCO dataset:
%
%                                            Mean = [123.6750 116.2800 103.5300]
%                                            StandardDeviation = [58.3950 57.1200 57.3750]
%
% yolov8ObjectDetector properties:
%   ModelName                    - Name of the trained object detector.
%   Network                      - YOLO v8 object detection network. (read-only)
%   ClassNames                   - A cell array of object class names. (read-only)
%   InputSize                    - Image size used during training. (read-only)
%
% yolov8ObjectDetector methods:
%   detect                       - Detect objects in an image.
%
% Example 1: Detect objects using pretrained YOLO v8 detector.
% ------------------------------------------------------------
% % Load the pretrained detector.
% detector = yolov8ObjectDetector();
%
% % Read test image.
% I = imread('highway.png');
%
% % Run detector.
% [bboxes, scores, labels] = detect(detector, I);
%
% % Display results.
% detectedImg = insertObjectAnnotation(I, 'Rectangle', bboxes, labels);
% figure, imshow(detectedImg)
%
% Example 2: Detect objects using 'yolov8m' pretrained model.
% -----------------------------------------------------------
% % Load the pretrained detector.
% detector = yolov8ObjectDetector('yolov8m');
%
% % Read test image.
% I = imread('highway.png');
%
% % Run detector.
% [bboxes, scores, labels] = detect(detector, I);
%
% % Display results.
% detectedImg = insertObjectAnnotation(I, 'Rectangle', bboxes, labels);
% figure imshow(detectedImg)
%
% Example 3: Configure a pretrained YOLO v8 detector for transfer learning.
% -------------------------------------------------------------------------
% % Specify the input image size.
% imageSize = [224 224 3];
%
% % Specify the class names.
% classes = {'car','person'};
%
% % Configure the detector.
% detector = yolov8ObjectDetector('yolov8s',classes,InputSize = imageSize);
%
% See also yoloxObjectDetector, trainYOLOv4ObjectDetector
%          yolov4ObjectDetector, imageLabeler.

% Copyright 2024 The MathWorks, Inc.

classdef yolov8ObjectDetector < vision.internal.detector.ObjectDetector
    properties(SetAccess = protected)
        % Network is a dlnetwork object with image input layer.
        Network
        
        % ClassNames specifies the names of the classes that YOLO v4 object
        % detector can detect.
        ClassNames
        
        % InputSize is a vector of the form [height width] or [height width channels]
        % defining image size used to train the detector. During detection, 
        % an input image is resized to this size before it is processed by
        % the detection network.
        InputSize
    end

    properties(Dependent = true)
        % NormalizationStatistics specifies z-score normalization statitics
        % as a structure with fields, Mean and StandardDeviation specified
        % as 1-by-C array of means and standard deviation per channel. The
        % number of channels, C must match the InputSize
        NormalizationStatistics
    end

    properties (Access = private, Hidden)
        NormalizationStatisticsInternal = [];
    end    

    methods
        function this = yolov8ObjectDetector(detectorInp,classNames,options)
            arguments
                detectorInp {isOneOrMoreType} = iGetSmallNetworkDetectorName();
                classNames = [];
                options.InputSize {mustBeNumeric, mustBePositive, mustBeReal, mustBeFinite, mustBeRGBSize} = []
                options.ModelName {mustBeTextScalar} = ""
                options.NormalizationStatistics = []
            end

            vision.internal.requiresNeuralToolbox(mfilename);
            options.customNetwork = isequal(class(detectorInp),'dlnetwork');
            
            if (~options.customNetwork && detectorInp == "uninitialized")
                return
            end

            % Loads and configure the pretrained model as specified in detectorName.
            params = yolov8ObjectDetector.parsePretrainedDetectorInputs(detectorInp,classNames,options);
            if options.customNetwork
                this.Network = detectorInp;
            else
                this.Network = iDownloadAndUpdatePretrainedModels(detectorInp, params);
            end

            this.InputSize = params.InputSize;
            if isempty(params.NormalizationStatistics)
                this.NormalizationStatistics = iDefaultNormalizationStats(this.InputSize(3));
            else
                this.NormalizationStatistics = params.NormalizationStatistics;
            end

            if ~isfield(params,"ClassNames")
                this.ClassNames = helper.getCOCOClassNames;
            else
                this.ClassNames = params.ClassNames;
            end
            this.InputSize = params.InputSize;
            this.ModelName = params.ModelName;

            this.Network = initialize(this.Network);
        end
    end

    methods
        function varargout = detect(detector, I, options)
            % bboxes = detect(detector,I) detects objects within the image I.
            % The location of objects within I are returned in bboxes, an
            % M-by-4 matrix defining M bounding boxes. Each row of bboxes
            % contains a four-element vector, [x, y, width, height]. This
            % vector specifies the upper-left corner and size of a bounding
            % box in pixels. detector is a yoloxObjectDetector object
            % and I is a truecolor or grayscale image.
            %
            % [..., scores] = detect(detector,I) optionally return the class
            % specific confidence scores for each bounding box. The scores
            % for each detection is product of objectness prediction and
            % classification scores. The range of the scores is [0 1].
            % Larger score values indicate higher confidence in the
            % detection.
            %
            % [..., labels] = detect(detector,I) optionally return the labels
            % assigned to the bounding boxes in an M-by-1 categorical
            % array. The labels used for object classes is defined during
            % training.
            %
            % detectionResults = detect(yolo,ds) detects objects within the
            % series of images returned by the read method of datastore,
            % ds. ds, must be a datastore that returns a table or a cell
            % array with the first column containing images.
            % detectionResults is a 3-column table with variable names
            % "Boxes", "Scores", and "Labels" containing bounding boxes,
            % scores, and the labels. The location of objects within an
            % image, I are returned in bounding boxes, an M-by-4 matrix
            % defining M bounding boxes. Each row of boxes contains a
            % four-element vector, [x, y, width, height]. This vector
            % specifies the upper-left corner and size of a bounding box in
            % pixels.
            %
            % [...] = detect(..., Name=Value) specifies additional
            % name-value pairs described below:
            %
            % "Threshold"              A scalar between 0 and 1. Detections
            %                          with scores less than the threshold
            %                          value are removed. Increase this value
            %                          to reduce false positives.
            %
            %                          Default: 0.25
            %
            % "SelectStrongest"        A logical scalar. Set this to true to
            %                          eliminate overlapping bounding boxes
            %                          based on their scores. This process is
            %                          often referred to as non-maximum
            %                          suppression. Set this to false if you
            %                          want to perform a custom selection
            %                          operation. When set to false, all the
            %                          detected bounding boxes are returned.
            %
            %                          Default: true
            %
            % "MiniBatchSize"          The mini-batch size used for processing a
            %                          large collection of images. Images are grouped
            %                          into mini-batches and processed as a batch to
            %                          improve computational efficiency. Larger
            %                          mini-batch sizes lead to faster processing, at
            %                          the cost of more memory.
            %
            %                          Default: 128
            %
            % "ExecutionEnvironment"   The hardware resources used to run the
            %                          YOLOX detector. Valid values are:
            %
            %                           "auto"      Use a GPU if it is available,
            %                                       otherwise use the CPU.
            %
            %                           "cpu"       Use the CPU.
            %
            %                           "gpu"       Use the GPU. To use a GPU,
            %                                       you must have Parallel
            %                                       Computing Toolbox(TM), and
            %                                       a CUDA-enabled NVIDIA GPU.
            %                                       If a suitable GPU is not
            %                                       available, an error message
            %                                       is issued.
            %
            %                          Default : "auto"
            %
            % "Acceleration"           Optimizations that can improve
            %                          performance at the expense of some
            %                          overhead on the first call, and possible
            %                          additional memory usage. Valid values
            %                          are:
            %
            %                           "auto"    - Automatically select
            %                                       optimizations suitable
            %                                       for the input network and
            %                                       environment.
            %
            %                           "mex"     - (GPU Only) Generate and
            %                                       execute a MEX function.
            %
            %                           "none"    - Disable all acceleration.
            %
            %                          Default : "auto"
            %
            % "AutoResize"             Logical scalar which specifies whether
            %                          or not the detect method automatically
            %                          resizes the input images to preserve
            %                          aspect ratio. When set to true, images
            %                          are resized to the nearest InputSize
            %                          by preserving the aspect ratio.
            %
            %                          Default: true
            %
            %  Notes:
            %  -----
            %  - When "SelectStrongest" is true the selectStrongestBboxMulticlass
            %    function is used to eliminate overlapping boxes. By
            %    default, the function is called as follows:
            %
            %   selectStrongestBboxMulticlass(bbox,scores,labels,...
            %                                       "RatioType", "Union", ...
            %                                       "OverlapThreshold", 0.45);
            %
            %  - When the input image size does not match the network input size, the
            %    detector resizes the input image to the detector.InputSize.
            %
            % Class Support
            % -------------
            % The input image I can be uint8, uint16, int16, double,
            % single, and it must be real and non-sparse.
            %
            % Example
            % -------
            % % Load pre-trained detector.
            % detector = yolov8ObjectDetector("yolov8s");
            %
            % % Read test image.
            % I = imread("kobi.png");
            %
            % % Run detector.
            % [bboxes, scores, labels] = detect(detector, I);
            %
            % % Display results.
            % detectedImg = insertObjectAnnotation(I, "Rectangle", bboxes, labels);
            % figure
            % imshow(detectedImg)
            arguments
                detector yolov8ObjectDetector
                I {mustBeA(I,["numeric","matlab.io.Datastore","matlab.io.datastore.Datastore","gpuArray"]),mustBeNonempty}
                options.Threshold (1,1) {yolov8ObjectDetector.checkThreshold(options.Threshold, 'yolov8ObjectDetector')} = 0.25
                options.SelectStrongest (1,1) {vision.internal.inputValidation.validateLogical(options.SelectStrongest, 'SelectStrongest')} = true
                options.MiniBatchSize (1,1) {vision.internal.cnn.validation.checkMiniBatchSize(options.MiniBatchSize, 'yolov8ObjectDetector')} = 128
                options.ExecutionEnvironment {mustBeMember(options.ExecutionEnvironment,{'gpu','cpu','auto'})} = "auto"
                options.Acceleration {mustBeMember(options.Acceleration,{'auto','mex','none'})} = "auto"
                options.AutoResize (1,1) {vision.internal.inputValidation.validateLogical(options.AutoResize, 'AutoResize')} = true
            end
            
            [params, networkInputSize] = validateImageInput(detector, I);

            params.SelectStrongest          = logical(options.SelectStrongest);
            params.MiniBatchSize            = double(options.MiniBatchSize);
            params.Threshold                = single(options.Threshold);
            params.NMSThreshold             = single(0.45);
            params.NetworkInputSize         = double(networkInputSize);
            params.ExecutionEnvironment     = string(options.ExecutionEnvironment);
            params.Acceleration             = string(options.Acceleration);
            params.AutoResize               = logical(options.AutoResize);

            [varargout{1:nargout}] = performDetect(detector, I, params);
        end

        function [bboxes, scores, labelIds] = detectYOLOv8(dlnet, image, numClasses, options)
            % detectYOLOv8 runs prediction on a trained yolov8 network.
            %
            % Inputs:
            % dlnet                - Pretrained yolov8 dlnetwork.
            % image                - RGB image to run prediction on. (H x W x 3)
            % numClasses           - Number of classes yolov8Detector is trained on.
            % executionEnvironment - Environment to run predictions on. Specify cpu,
            %                        gpu, or auto.
            %
            % Outputs:
            % bboxes     - Final bounding box detections ([x y w h]) formatted as
            %              NumDetections x 4.
            % scores     - NumDetections x 1 classification scores.
            % labelIds   - NumDetections x 1 label Ids.

            arguments
                dlnet dlnetwork
                image {mustBeA(image,["numeric","gpuArray"]),mustBeNonempty}
                numClasses (1,1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger, mustBeFinite, mustBeNonsparse, mustBeNonempty}
                options.ExecutionEnvironment {mustBeMember(options.ExecutionEnvironment,{'gpu','cpu','auto'})} = "auto"
            end            

            % Get the input size of the network.
            inputSize = dlnet.Layers(1).InputSize;

            % Apply Preprocessing on the input image.
            origSize = size(image);
            Ibgr = image(:,:,[3,2,1]); % convert image to bgr
            img = helper.preprocess(Ibgr, inputSize);

            newSize = size(img);
            img = img(:,:,[3,2,1]); % convert image to rgb

            % Convert to dlarray.
            dlInput = dlarray(img, 'SSCB');

            % If GPU is available, then convert data to gpuArray.
            if (options.ExecutionEnvironment == "auto" && canUseGPU) || options.ExecutionEnvironment == "gpu"
                dlInput = gpuArray(dlInput);
            end

            % Perform prediction on the input image.
            outFeatureMaps = cell(length(dlnet.OutputNames), 1);
            [outFeatureMaps{:}] = predict(dlnet, dlInput);

            % Apply postprocessing on the output feature maps.
            [bboxes,scores,labelIds] = helper.postprocess(outFeatureMaps, ...
                origSize, newSize, numClasses);

            bboxes = gather(bboxes);
            scores = gather(scores);
            labelIds = gather(labelIds);

        end
    end

    %----------------------------------------------------------------------
    methods

        function this = set.NormalizationStatistics(this,statsStruct)

            iValidateNormalizationStats(statsStruct,this.InputSize(3))
            
            this.NormalizationStatisticsInternal = struct("Mean",gather(reshape(statsStruct.Mean,[1 this.InputSize(3)])),...
                "StandardDeviation",gather(reshape(statsStruct.StandardDeviation,[1 this.InputSize(3)])));
            
            statsStructForInputNorm = statsStruct;
            if ~all(isfield(statsStructForInputNorm, {'Mean','Std','Max','Min'}))
                statsStructForInputNorm.Mean = reshape(statsStructForInputNorm.Mean,[1 1 this.InputSize(3)]);
                statsStructForInputNorm.Std = reshape(statsStructForInputNorm.StandardDeviation,[1 1 this.InputSize(3)]);
                statsStructForInputNorm.Min = [];
                statsStructForInputNorm.Max = [];
            end

            this = setInputNormalization(this,statsStructForInputNorm);
        end

        function statsStruct = get.NormalizationStatistics(this)
            statsStruct = this.NormalizationStatisticsInternal;
        end

    end

    %----------------------------------------------------------------------
    methods(Hidden)
        %------------------------------------------------------------------
        % Preprocess input data.
        %------------------------------------------------------------------
        function varargout = preprocess(detector, I, varargin)
            % This method preprocesses the input data prior to calling
            % the predict method. It resizes the input data to the
            % detector.InputSize when params.AutoResize is false.
            % Otherwise, input data is passed as-is.

            params = parsePreprocessInputs(detector, I, varargin);
            if params.DetectionInputIsDatastore
                % Copy and reset the given datastore, so external state events are
                % not reflected.
                ds = copy(I);
                reset(ds);

                fcn = @iPreprocessForDetect;
                % We need just the preprocessed image -> num arg out is 1.
                fcnArgOut = 2;
                varargout{1} = transform(ds, @(x)iPreProcessForDatastoreRead(x,fcn,fcnArgOut,...
                    params.ExecutionEnvironment,detector.InputSize,...
                    params.AutoResize,params.CastToGpuArray));
                varargout{2} = {};
            else
                [varargout{1:nargout}] = iPreprocessForDetect(I, ...
                    params.ExecutionEnvironment,detector.InputSize,...
                    params.AutoResize,params.CastToGpuArray);
            end
        end

        %------------------------------------------------------------------
        % Predict output feature maps.
        %------------------------------------------------------------------
        function outputFeatures = predict(detector,dlX,varargin)
            % This method predicts features of the preprocessed image dlX.
            % The outputFeatures is a N-by-1 cell array, where N are the
            % number of outputs in network. Each cell of outputFeature
            % contains predictions from an output layer.

            predictParams = parsePredictInputs(detector,varargin);
            network = detector.Network;
            if (~isnumeric(dlX) && ~iscell(dlX))

                % Process datastore with network and output the predictions.
                loader = iCreateDataLoader(dlX,predictParams.MiniBatchSize,predictParams.NetworkInputSize);

                % Iterate through data and write results to disk.
                k = 1;

                bboxes = cell(predictParams.MiniBatchSize, 1);
                scores = cell(predictParams.MiniBatchSize, 1);
                labels = cell(predictParams.MiniBatchSize, 1);

                while hasdata(loader)
                    X = nextBatch(loader);
                    imgBatch = X{1};
                    batchInfo = X{2};
                    numMiniBatch = size(batchInfo,1);

                    % Compute predictions.
                    features = iPredictActivations(network, imgBatch, predictParams.Acceleration);

                    for ii = 1:numMiniBatch
                        for outIdx = 1:size(network.OutputNames,2)
                            fmap{outIdx,1} = features{outIdx,1}(:,:,:,ii);
                        end
                        [bboxes{k,1},scores{k,1},labels{k,1}] = ...
                            postprocess(detector,fmap, batchInfo{ii}, varargin{1,1});
                        k = k + 1;
                    end
                end
                outputFeatures = cell(1,3);
                outputFeatures{1,1} = bboxes(1:k-1);
                outputFeatures{1,2} = scores(1:k-1);
                outputFeatures{1,3} = labels(1:k-1);

            else
                if iscell(dlX)
                    outputFeatures = iPredictBatchActivations(network, dlX, predictParams.Acceleration);
                else
                    if size(dlX,4)>1
                        outputFeatures = iPredictMultiActivations(network, dlX, predictParams.Acceleration);
                    else
                        outputFeatures = iPredictActivations(network, dlX, predictParams.Acceleration);
                    end
                end
            end
        end

        %------------------------------------------------------------------
        % Postprocess output feature maps.
        %------------------------------------------------------------------
        function varargout = postprocess(detector,YPredData, info, params)
            % This method applies post-processing on the predicted output
            % feature maps and computes the detected bounding boxes, scores
            % and labels.

            if isequal(size(YPredData), [1,3])
                varNames = {'Boxes', 'Scores', 'Labels'};
                result = table(YPredData{1,1}, YPredData{1,2}, YPredData{1,3}, 'VariableNames', varNames);
                [varargout{1:nargout}] = result;
            else
                if params.DetectionInputWasBatchOfImages
                    [varargout{1:nargout}] = iPostprocessMultiDetection(detector,YPredData,info);
                else
                    [varargout{1:nargout}] = iPostprocessSingleDetection(detector,YPredData,info);
                end
            end
        end

        %------------------------------------------------------------------
        % Parse preprocess input parameters.
        %------------------------------------------------------------------
        function params = parsePreprocessInputs(~, I, varargin)
            params.AutoResize = varargin{1,1}{1,1}.AutoResize;
            params.ExecutionEnvironment = varargin{1,1}{1,1}.ExecutionEnvironment;
            params.DetectionInputIsDatastore = ~isnumeric(I) && ~iscell(I);
            params.CastToGpuArray = varargin{1,1}{1,1}.CastToGpuArray; 
        end

        %------------------------------------------------------------------
        % Parse predict input parameters.
        %------------------------------------------------------------------
        function params = parsePredictInputs(~,varargin)
            params = varargin{1,1}{1,1};
        end
    end

    methods(Static, Hidden, Access = protected)
        %------------------------------------------------------------------
        % Parse and validate pretrained detector parameters.
        %------------------------------------------------------------------
        function params = parsePretrainedDetectorInputs(detectorInp,classNames,options)
            % Parse inputs for this syntax:
            % detector = yoloxObjectDetector(detectorName).

            params = options;
            
            if ~isequal(class(detectorInp),'dlnetwork')
                params.DetectorName = detectorInp;
                inputSize = [640 640 3];
            else
                params.DetectorName = 'custom';
                inputSize = detectorInp.Layers(1,1).InputSize;
            end

            % Parse inputs for this syntax:
            % detector = yoloxObjectDetector(detectorName,classNames).
            if isempty(classNames) && ~isempty(options.InputSize)
                error('classNames must be specified to configure detector for training using InputSize');
            end

            if ~isempty(classNames)
                params.ClassNames = classNames;

                if ~iscolumn(params.ClassNames)
                    params.ClassNames = params.ClassNames';
                end
                if isstring(params.ClassNames) || iscategorical(params.ClassNames)
                    params.ClassNames = cellstr(params.ClassNames);
                end

                iValidateClassNames(params.ClassNames);
            end

            if isempty(options.InputSize)
                params.InputSize = inputSize;
                params.UpdateInputLayer = false;
            else
                params.UpdateInputLayer = true;
            end

            iCheckInputSize(params.InputSize);

            if params.InputSize(1) == 1 || params.InputSize(2) == 1
                error(message('visualinspection:yoloxObjectDetector:inputSizeMustBeAtleastTwo',params.DetectorName));
            end

            if strcmp(params.ModelName,"")
                params.ModelName = params.DetectorName;
            end

            if ~isempty(params.NormalizationStatistics)
                iValidateNormalizationStats(params.NormalizationStatistics,params.InputSize(3))
            end
            
        end        
    end

    methods (Hidden)
        function this = setInputNormalization(this,stats)
            network = this.Network;
            currentInputLayer = this.Network.Layers(1);
            map = normalizationStatsDictionary(stats);
            statsSet = map(currentInputLayer.Normalization);
            inputSize = size(getExampleInputsFromNetwork(this.Network));
            newInputLayer = imageInputLayer(inputSize,"Name",currentInputLayer.Name,...
                "Normalization",currentInputLayer.Normalization,...
                statsSet{:});
            network = replaceLayer(network,this.Network.Layers(1).Name,newInputLayer);
            this.Network = initialize(network);
        end
    end

    %======================================================================
    % Save/Load
    %======================================================================
    methods(Hidden)
        function s = saveobj(this)
            s.Version                      = 1.0;
            s.ModelName                    = this.ModelName;
            s.Network                      = this.Network;
            s.ClassNames                   = this.ClassNames;
            s.InputSize                    = this.InputSize;
            s.NormalizationStatistics      = this.NormalizationStatistics;
        end

        function dlnet = matlabCodegenPrivateNetwork(this)
            dlnet = dlnetwork(this);
        end
    end

    methods(Static, Hidden)
        function this = loadobj(s)
            try
                vision.internal.requiresNeuralToolbox(mfilename);
                this = yolov8ObjectDetector("uninitialized");
                this.Network                 = s.Network;
                this.ClassNames              = s.ClassNames;
                this.InputSize               = s.InputSize;
                this.ModelName               = s.ModelName;
                this.NormalizationStatistics = s.NormalizationStatistics;

            catch ME
                rethrow(ME)
            end
        end
    end

    %----------------------------------------------------------------------
    methods(Static, Hidden)
        function data = preprocessInput(data, targetSize)
            batchSize = size(data,4);
            if batchSize>1
                dataTmp = [];
                for i = 1:batchSize
                    Itmp = helper.preprocess(data(:,:,:,i),targetSize(1:2));
                    if isempty(dataTmp)
                        dataTmp = Itmp;
                    else
                        dataTmp = cat(4,dataTmp,Itmp);
                    end
                end
                data = dataTmp;
            else
                data = helper.preprocess(data, targetSize(1:2));
            end
        end
    end
end

%--------------------------------------------------------------------------
function isOneOrMoreType(detectorInp)
if ~isempty(detectorInp)
    detectorInp = convertCharsToStrings(detectorInp);
    if isstring(detectorInp)
        tf = ismember(detectorInp, {'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x', 'uninitialized'});
    else
        tf = isequal(class(detectorInp),'dlnetwork');
    end
    assert(tf,'Input must be either a supported YOLOv8 detector name or a trained YOLOv8 network');
end
end

%--------------------------------------------------------------------------
function detectorName = iGetSmallNetworkDetectorName()
detectorName = "yolov8s";
end

%--------------------------------------------------------------------------
function iCheckInputSize(inputSize)
validateattributes(inputSize, {'numeric'}, ...
    {'2d','nonempty','nonsparse',...
    'real','finite','integer','positive','nrows',1,'numel',3});

if any(mod(inputSize(1:2),32))
    error('Height and Width of input size [H W C] should be factor of 32');
end

end

%--------------------------------------------------------------------------
function iValidateClassNames(value)
if ~isvector(value) || ~iIsValidDataType(value)
    error('Classes must be specified as a string vector, a cell array of character vectors, or a categorical vector.');
end
if iHasDuplicates(value)
    error('Classes must be unique.');
end
end

%--------------------------------------------------------------------------
function outputFeatures = iPredictMultiActivations(network,dlX, acceleration)
    numMiniBatch = size(dlX,4);
    outputFeatures = cell(numMiniBatch,1);
    for ii = 1:numMiniBatch
        inp = dlX(:,:,:,ii);
        outputFeatures{ii,1} = iPredictActivations(network, inp, acceleration);
    end
end

%--------------------------------------------------------------------------
function outputFeatures = iPredictBatchActivations(network,dlX, acceleration)
    numMiniBatch = size(dlX,2);
    outputFeatures = cell(numMiniBatch,1);
    for ii = 1:numMiniBatch
        inp = dlX{ii};
        outputFeatures{ii,1} = iPredictActivations(network, inp, acceleration);
    end
end

function outputFeatures = iPredictActivations(network, dlX, acceleration)
% Perform prediction on the input image.
outputFeatures = cell(length(network.OutputNames), 1);
[outputFeatures{:}] = predict(network, dlX,'Acceleration',acceleration);
end
%--------------------------------------------------------------------------
function [bboxes,scores,labels] = iPostprocessMultiDetection(detector,YPredData,info)
    numMiniBatch = size(YPredData,1);
    bboxes = cell(numMiniBatch, 1);
    scores = cell(numMiniBatch, 1);
    labels = cell(numMiniBatch, 1);
    for ii = 1:numMiniBatch
        [bboxes{ii},scores{ii},labels{ii}] = ...
            iPostprocessSingleDetection(detector,YPredData{ii,1},info);
    end
end

%--------------------------------------------------------------------------
function [bboxes,scores,labels] = iPostprocessSingleDetection(detector,YPredData,info)
    % Obtain the classnames detector is trained on.
    classes = detector.ClassNames;
    numClasses = size(classes,1);

    % Apply postprocessing on the output feature maps.
    [bboxes,scores,labelIds] = helper.postprocess(YPredData, ...
        info.InputImageSize, info.ProcessedImageSize, numClasses);

    bboxes = gather(bboxes);
    scores = gather(scores);
    labelIds = gather(labelIds);

    labels = categorical(classes,cellstr(classes));
    labels = labels(labelIds);
end

%--------------------------------------------------------------------------
function tf = iIsValidDataType(value)
    tf = iscategorical(value) || iscellstr(value) || isstring(value);
end

%--------------------------------------------------------------------------
function tf = iHasDuplicates(value)
    tf = ~isequal(value, unique(value, 'stable'));
end

%--------------------------------------------------------------------------
function out = iPreProcessForDatastoreRead(in, fcn, numArgOut, varargin)
    if isnumeric(in)
        % Numeric input
        in = {in};
    end
    if istable(in)
        % Table input
        in = in{:,1};
    else
        % Cell input
        in = in(:,1);
    end
    numItems = numel(in);
    out = cell(numItems, numArgOut);
    for ii = 1:numel(in)
        [out{ii, 1:numArgOut}] = fcn(in{ii},varargin{:});
    end
end

%--------------------------------------------------------------------------
function [Ipreprocessed,info] = iPreprocessForDetect(I, executionEnvironment, networkInputSize, autoResize, castToGpuArray)

    % Check if the input datatype is valid or not.
    if ~(isa(I,'uint8') || isa(I,'uint16') || isa(I,'int16') || ...
            isa(I,'double') || isa(I,'single') || isa(I,'gpuArray'))
        error('Input datatype must be uint8, uint16, int16, double or single.');
    end

    % Convert to gpuArray based on executionEnvironment.
    if castToGpuArray
        if (strcmp(executionEnvironment,'auto') && canUseGPU) || strcmp(executionEnvironment,'gpu')
            I = gpuArray(I);
        end
    end

    % Save preprocessed image size.
    info.PreprocessedImageSize = networkInputSize;

    % Compute scale factors to scale boxes from targetSize back to the input size.
    if autoResize
        Ipreprocessed = yolov8ObjectDetector.preprocessInput(I, info.PreprocessedImageSize);
        info.InputImageSize = size(I);
        info.ProcessedImageSize = size(Ipreprocessed);
    else
        sz = size(I);
        scale   = sz(1:2)./networkInputSize(1:2);
        [~,idx] = max(scale);
        [info.ScaleX,info.ScaleY] = deal(scale(idx),scale(idx));
        info.InputImageSize = sz;

        Ipreprocessed = single(I);
    end

    Ipreprocessed = dlarray(Ipreprocessed,'SSCB');
end

%--------------------------------------------------------------------------
function loader = iCreateDataLoader(ds,miniBatchSize,inputLayerSize)
    loader = nnet.internal.cnn.DataLoader(ds,...
        'MiniBatchSize',miniBatchSize,...
        'CollateFcn',@(x)iTryToBatchData(x,inputLayerSize));
end

%--------------------------------------------------------------------------
function data = iTryToBatchData(X, inputLayerSize)
    try
        observationDim = numel(inputLayerSize) + 1;
        data{1} = cat(observationDim,X{:,1});
    catch e
        if strcmp(e.identifier, 'MATLAB:catenate:dimensionMismatch')
            error(message('visualinspection:yoloxObjectDetector:unableToBatchImagesForDetect'));
        else
            throwAsCaller(e);
        end
    end
    data{2} = X(:,2:end);
end

%--------------------------------------------------------------------------
function network = iDownloadAndUpdatePretrainedModels(modelName, params)
data = downloadPretrainedYOLOv8 (modelName);
network = data.yolov8Net;
if params.UpdateInputLayer
    network = iUpdateFirstConvChannelsAndInputLayer(network,params.InputSize);
end
end

%--------------------------------------------------------------------------
function model = downloadPretrainedYOLOv8(modelName)
% The downloadPretrainedYOLOv8 function downloads a YOLO v8 network 
% pretrained on COCO dataset.
%
% Copyright 2024 The MathWorks, Inc.

supportedNetworks = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"];
validatestring(modelName, supportedNetworks);

modelName = convertContainedStringsToChars(modelName);

netMatFileFullPath = fullfile(pwd, [modelName, '.mat']);

if ~strcmp(modelName,'yolov8n')
    if ~exist(netMatFileFullPath,'file')
        fprintf(['Downloading pretrained ', modelName ,' network.\n']);
        fprintf('This can take several minutes to download...\n');
        url = ['https://github.com/matlab-deep-learning/Pretrained-YOLOv8-Network-For-Object-Detection/releases/download/v1.0.0/', modelName, '.mat'];
        websave(netMatFileFullPath, url);
        fprintf('Done.\n\n');
    else
        fprintf(['Pretrained ', modelName, ' network already exists.\n\n']);
    end
end

model = load(netMatFileFullPath);
end

%--------------------------------------------------------------------------
function dlnetOut = iUpdateFirstConvChannelsAndInputLayer(dlnet,imageSize)
    % This function update the channels of first conv layer if InputSize channel
    % does not match with channels of first conv layer. It also updates the
    % imageInputLayer or initializes the dlnetwork if image input layer not present.

    if size(imageSize,2)==2
        imageSize = [imageSize 1];
    end

    outputNames = dlnet.OutputNames;

    imgIdx = arrayfun(@(x)isa(x,'nnet.cnn.layer.ImageInputLayer'),...
        dlnet.Layers);
    imageInputIdx = find(imgIdx,1,'first');

    numChannel = imageSize(3);

    idx = arrayfun(@(x)isa(x,'nnet.cnn.layer.Convolution2DLayer'),...
        dlnet.Layers);
    convIdx = find(idx,1,'first');
    if ~isempty(convIdx)
        numFirstConvLayerChannels = dlnet.Layers(convIdx,1).NumChannels;
    else
        error('Input network must have convolution layer');
    end

    needToReplaceFirstConvLayer = ~strcmp(numFirstConvLayerChannels,'auto') && ...
        numFirstConvLayerChannels~=numChannel;
    needToReplaceInputLayer = ~isempty(imageInputIdx);

    if needToReplaceFirstConvLayer || needToReplaceInputLayer
        % Capture the layers once if we're going to edit the graph
        layers = dlnet.Layers;
    end
    
    % If number of channels in imageSize is not equal to the channel count
    % of first convolutional layer, update the channel count of first conv
    % layer and use values of properties as it is. Pyramid pooling concept
    % has been used for concatenating extra channel. Each extra channel is
    % mean of original (initial) channels of conv layer.
    %
    % Zhao, Hengshuang, et al. "Pyramid Scene Parsing Network." 2017 IEEE
    % Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2017.
    if needToReplaceFirstConvLayer
        firstConvLayer = layers(convIdx,1);
        firstConvLayerWeights = firstConvLayer.Weights;
        % Average over the RGB values while maintaining the space to
        % depth tiling of the spatial dimension
        channelIndicesToAverage = [1 5 9];
        meanChannelWeightTile1  = reshape(mean(firstConvLayerWeights(:,:,channelIndicesToAverage,:),3),size(firstConvLayerWeights(:,:,1,:)));
        meanChannelWeightsTile2 = reshape(mean(firstConvLayerWeights(:,:,channelIndicesToAverage+1,:),3),size(firstConvLayerWeights(:,:,1,:)));
        meanChannelWeightsTile3 = reshape(mean(firstConvLayerWeights(:,:,channelIndicesToAverage+2,:),3),size(firstConvLayerWeights(:,:,1,:)));
        meanChannelWeightsTile4 = reshape(mean(firstConvLayerWeights(:,:,channelIndicesToAverage+3,:),3),size(firstConvLayerWeights(:,:,1,:)));
        meanChannelWeights = cat(3,meanChannelWeightTile1,meanChannelWeightsTile2,meanChannelWeightsTile3,meanChannelWeightsTile4);
        if numChannel>numFirstConvLayerChannels
            extraChanels = abs(numChannel-numFirstConvLayerChannels);
            extraChannelWeights = repmat(meanChannelWeights,1,1,extraChanels);
            updatedConvLayerWeights = cat(3,firstConvLayerWeights,extraChannelWeights);
        else
            updatedConvLayerWeights = repmat(meanChannelWeights,1,1,numChannel);
        end
        updatedConvLayer = convolution2dLayer(...
            firstConvLayer.FilterSize, firstConvLayer.NumFilters, ...
            'NumChannels', 4*numChannel, ...
            'Stride',firstConvLayer.Stride,...
            'Padding',firstConvLayer.PaddingSize , ...
            'PaddingValue',firstConvLayer.PaddingValue,...
            'DilationFactor', firstConvLayer.DilationFactor, ...
            'Weights',updatedConvLayerWeights,...
            'Bias',firstConvLayer.Bias,...
            'WeightL2Factor',firstConvLayer.WeightL2Factor,...
            'BiasL2Factor',firstConvLayer.BiasL2Factor,...
            'WeightLearnRateFactor',firstConvLayer.WeightLearnRateFactor,...
            'BiasLearnRateFactor',firstConvLayer.BiasLearnRateFactor,...
            'Name', firstConvLayer.Name, ...
            'WeightsInitializer', firstConvLayer.WeightsInitializer, ...
            'BiasInitializer', firstConvLayer.BiasInitializer);
        dlnet = replaceLayer(dlnet,layers(convIdx).Name,...
            updatedConvLayer);
    end

    % If imageSize is not equal to the InputSize, replace the imageInputLayer.
    if needToReplaceInputLayer
        inputLayer = layers(imageInputIdx,1);
        if ~isequal(inputLayer.InputSize,imageSize)
            imageInput = imageInputLayer(imageSize, ...
                'Normalization','zscore', ...
                'Mean',layers(imageInputIdx).Mean,...
                'StandardDeviation',layers(imageInputIdx).StandardDeviation,...
                'Name',layers(imageInputIdx).Name);
            dlnet = replaceLayer(dlnet,layers(imageInputIdx).Name,...
                imageInput);
        end
    end
    dlX = dlarray(rand(imageSize, 'single'), 'SSCB');
    dlnetOut = initialize(dlnet,dlX);
    dlnetOut.OutputNames = outputNames;
end

function map = normalizationStatsDictionary(stats)
    % This maps knowledge of how different styles of normalization in the input
    % layer (Keys) map to different Name/Value inputs to the statistics field
    % of the input layer.
    map = containers.Map({'zerocenter','zscore','rescale-symmetric','rescale-zero-one','none'},...
        { {'Mean',gather(stats.Mean)}, {'StandardDeviation',gather(stats.Std),'Mean',gather(stats.Mean)},...
        { 'Min', gather(stats.Min), 'Max', gather(stats.Max) },...
        { 'Min', gather(stats.Min), 'Max', gather(stats.Max) },...
        {} });

end

function iValidateNormalizationStats(stats,inputChannelSize)
mustBeA(stats,"struct");
mustBeScalarOrEmpty(stats)
tf = isfield(stats, {'Mean','StandardDeviation'});
if ~all(tf)
    error('The NormalizationStatistics structure must contain the following fields: Mean and StandardDeviation.');
end

meanSize = size(stats.Mean);
stdSize = size(stats.StandardDeviation);
iValidateNormalizationStatsSize(meanSize,inputChannelSize);
iValidateNormalizationStatsSize(stdSize,inputChannelSize);

end

function iValidateNormalizationStatsSize(statsSize,inputChannelSize)
if (numel(statsSize) == 2 && any(statsSize ~= [1 inputChannelSize])) || ...
        (numel(statsSize) == 3 && any(statsSize ~= [1 1 inputChannelSize])) || ...
        numel(statsSize) > 3
    error(message('visualinspection:yoloxObjectDetector:invalidNormalizationStatisticsSize',inputChannelSize));
end
end

function x = iCreateDummyInput(inputSize)
    dims = repmat('S',1,numel(inputSize)-1);
    dims = [dims,'C'];
    x = dlarray(zeros(inputSize),dims);
end

function x = getExampleInputsFromNetwork(net)
    x = getExampleInputs(net); % Populated when a user calls initialize on network without input layer.
    if isempty(x)
        inputSize = net.Layers(1).InputSize;
        x = iCreateDummyInput(inputSize);
    else
        x = iCreateDummyInput(size(x{1},[1 2 3]));
    end
end

function [params, networkInputSize] = validateImageInput(detector, I)

params.DetectionInputIsDatastore = ~isnumeric(I);

if params.DetectionInputIsDatastore
    sampleImage = vision.internal.cnn.validation.checkDetectionInputDatastore(I, mfilename);
else
    if ndims(I) > 4
        error('Input numeric data cannot have more than 4 dimensions');
    end
    sampleImage = I;    
end
params.CastToGpuArray = ~isgpuarray(sampleImage);

if size(detector.InputSize,2) == 2
    networkInputSize = [detector.InputSize 1];
else
    networkInputSize = detector.InputSize;
end

validateChannelSize = true;  % check if the channel size is equal to that of the network input channel size
validateImageSize   = false; % yolox can support images smaller than input size
[~,params.DetectionInputWasBatchOfImages] = vision.internal.cnn.validation.checkDetectionInputImage(...
    networkInputSize,sampleImage,validateChannelSize,validateImageSize);
end

function statsStruct = iDefaultNormalizationStats(inputChannelSize)
if inputChannelSize == 3
    statsStruct = struct("Mean",[123.6750 116.2800 103.5300],"StandardDeviation",[58.3950   57.1200   57.3750]);
else
    avgMean = mean([123.6750 116.2800 103.5300]);
    avgStd = mean([58.3950   57.1200   57.3750]);
    avgMean = repmat(avgMean,1,inputChannelSize);
    avgStd = repmat(avgStd,1,inputChannelSize);
    statsStruct = struct("Mean",avgMean,"StandardDeviation",avgStd);
end
end

function mustBeRGBSize(input)    
     % the size must either be [] or (1,3) with the channel dim=3
        
     if(~isempty(input)&&length(input)>2)        
        isValidChannelDim = length(input)==3 && input(3)==3;
     else
         isValidChannelDim = false;
     end
     
     if~(isempty(input) || (isValidChannelDim))
        throwAsCaller(MException('visualinspection:yoloxObjectDetector:incorrectInputSize',...
           vision.getMessage('visualinspection:yoloxObjectDetector:incorrectInputSize')));
     end
end
