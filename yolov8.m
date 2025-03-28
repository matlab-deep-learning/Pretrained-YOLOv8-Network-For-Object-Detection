% yolov8 Create a YOLO V8 network for instance segmentation.
%
% detector = yolov8(detectorName) loads a pretrained YOLO V8 instance
% segmentation detector trained on the COCO dataset. The detectorName
% specifies the architecture of the pre-trained network. detectorName must
% be either 'yolov8n', 'yolov8s', 'yolov8n', 'yolov8l', 'yolov8x'.
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
%                               segmentation quality.
%
%                   'yolov8m'   Use this model for higher accuracy with
%                               moderate computational demands.
%
%                   'yolov8l'   Use this model to prioritize maximum
%                               segmentation accuracy for high-end systems,
%                               at the cost of computational intensity.
%
%                   'yolov8x'   Use this model to get most accurate
%                               segmentation but requires significant
%                               computational resources, ideal for high-end
%                               systems prioritizing segmentation
%                               performance.
%
% Additional input arguments
% ----------------------------
% [...] = yolov8(..., Name=Value) specifies additional name-value pair
% arguments to configure the pre-trained YOLO v8 network as described below:
%
%    "ModelName"       Detector name specified as string or character
%                      vector.
%
%                      Default: detectorName or specified detectorName
%
%
%    "InputSize"       Specify the image sizes to use for detection. The
%                      segmentObjects method resizes input images
%                      to this size in the detector while maintaining the
%                      aspect ratio.
%
%                      Default: network input size
%
% yolov8 object properties
% --------------------------
%   ModelName               - Name of the trained yolov8 network.
%   Network                 - YOLO v8 instance segmentation network. (read-only)
%   ClassNames              - A string array of object class names. (read-only)
%   InputSize               - The image size used during training. (read-only)
%
% yolov8 object methods
% -----------------------
%   segmentObjects -  Segment object instances in an image.
%
% Example - Segment instances using pre-trained YOLO V8
% -------------------------------------------------------
%
% I = imread('visionteam.jpg');
%
% % Load pre-trained YOLO v8 network
% detector = yolov8('yolov8m');
%
% % Run inference on the YOLO v8 network
% [masks,labels,scores] = segmentObjects(detector,I);
%
% % Visualize the results
% % Overlay the object masks on the input image
% overlayedImage = insertObjectMask(I, masks);
% figure, imshow(overlayedImage)
%
% See also segmentObjects, solov2, maskrcnn, evaluateInstanceSegmentation,
%          insertObjectMask.

% Copyright 2025 The MathWorks, Inc.

classdef yolov8

    % Publicly visible YOLO v8 properties
    properties(SetAccess=protected)
        % Network is a dlnetwork object with image input layer.
        Network

        % Custom model name
        ModelName

        % Class names the network is trained on
        ClassNames

        % Image Size which the network is trained on
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

    methods(Access=public)

        function obj = yolov8(detector, classNames,options)
            arguments
                detector {isOneOrMoreType} = iGetSmallNetworkDetectorName();
                classNames = [];
                options.InputSize {mustBeNumeric, mustBePositive, mustBeReal, mustBeFinite, mustBeRGBSize} = []
                options.ModelName {mustBeTextScalar} = ""
                options.NormalizationStatistics = []
            end

            vision.internal.requiresNeuralToolbox(mfilename);
            options.customNetwork = isequal(class(detector),'dlnetwork');

            % This is added to support load network workflows all the
            % weights and properties will be populated by the loadobj
            % method.
            if (~options.customNetwork && detector == "uninitialized")
                return;
            end

            % Loads and configure the pretrained model as specified in detectorName.
            params = yolov8.parsePretrainedDetectorInputs(detector,classNames,options);
            if options.customNetwork
                obj.Network = detector;
            else
                obj.Network = iDownloadAndUpdatePretrainedModels(detector, params);
            end

            obj.InputSize = params.InputSize;
            if isempty(params.NormalizationStatistics)
                obj.NormalizationStatistics = iDefaultNormalizationStats(obj.InputSize(3));
            else
                obj.NormalizationStatistics = params.NormalizationStatistics;
            end

            if ~isfield(params,"ClassNames")
                obj.ClassNames = helper.getCOCOClassNames;
            else
                obj.ClassNames = params.ClassNames;
            end

            obj.InputSize = params.InputSize;
            obj.ModelName = params.ModelName;

            obj.Network = initialize(obj.Network);
        end
    end

    methods(Static, Hidden, Access = protected)
        %------------------------------------------------------------------
        % Parse and validate pretrained segmentor parameters.
        %------------------------------------------------------------------
        function params = parsePretrainedDetectorInputs(detectorInp,classNames,options)
            % Parse inputs for this syntax:
            % detector = yolov8(detectorName).

            params = options;

            if ~isequal(class(detectorInp),'dlnetwork')
                params.DetectorName = detectorInp;
                inputSize = [640 640 3];
            else
                params.DetectorName = 'custom';
                inputSize = detectorInp.Layers(1,1).InputSize;
            end

            % Parse inputs for this syntax:
            % detector = yolov8(detectorName,classNames).
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

    methods(Access=public)
        function varargout = segmentObjects(obj, im, options)

            %SEGMENTOBJECTS Segment objects in an image using YOLO v8
            % instance segmentation network.
            %
            % masks = segmentObjects(yolov8Obj,I) returns object masks
            % within the image I using the trained yolov8 network. The
            % objects masks are detected as a H-by-W-by-M logical array,
            % where each channel contains the mask for a single object. H
            % and W are the height and width of the input image I and M is
            % the number of objects detected in the image. yolov8Obj is an
            % object of the yolov8 class and I is an RGB or grayscale
            % image.
            %
            % masks = segmentObjects(yolov8Obj,IBatch) returns objects
            % within each image contained in the batch of images IBatch.
            % IBatch is a numeric array containing images in the format
            % H-by-W-by-C-by-B, where B is the number of images in the
            % batch, and C is the channel size. For grayscale images, C
            % must be 1. masks is a B-by-1 cell array, with each cell
            % containing an H-by-W-by-M array of object masks for each
            % image in the batch, B.
            %
            % [masks, labels] = segmentObjects(yolov8Obj,I) optionally
            % returns the labels assigned to the detected M objects as an
            % M-by-1 categorical array. labels is a B-by-1 cell array, if
            % the input I is a batch of images of the format
            % H-by-W-by-C-by-B.
            %
            % [masks, labels, scores] = segmentObjects(yolov8Obj,I)
            % optionally return the detection scores for each of the M
            % objects. The score for each detection is the product of the
            % classification score and maskness score. The range of the
            % score is [0 1]. Larger score values indicate higher
            % confidence in the detection. scores is a B-by-1 cell array,
            % if the input I is a batch of images of the format
            % H-by-W-by-C-by-B.
            %
            % [masks, labels, scores, boxes] = segmentObjects(yolov8Obj,I)
            % optionally return location of objects within I in M-by-4
            % matrix defining M bounding boxes. Each row of axis-aligned
            % bboxes contain a four-element vector, [x, y, width, height].
            % This vector specifies the upper-left corner and size of a
            % bounding box in pixels.
            %
            % outds = segmentObjects(yolov8Obj, imds,___) returns the
            % instance segmentation results for images stored in a
            % Datastore imds. The output of read(ds) must be an image array
            % or a cell array. When the output of read(ds) is a cell array,
            % then only the first column of data is processed by the
            % network. The output outds is a fileDatastore representing the
            % instance segmentation results.  The result for each image -
            % object masks, labels, scores and bounding boxes are stored in
            % a .MAT file at the location specified by 'WriteLocation'. A
            % read on the outds returns these outputs in the following
            % order:
            %
            %       1st cell  : Predicted logical object masks for M objects as a
            %                   H-by-W-by-M logical array.
            %       2nd cell  : Predicted object labels as a Mx1 categorical vector.
            %
            %       3rd cell  : Prediction scores as a Mx1 numeric vector.
            %
            %       4th cell  : Prediction boxes as a Mx4 numeric vector.
            %
            % [...] = segmentObjects(..., Name=Value) specifies additional
            % name-value pairs described below:
            %
            % 'Threshold'              A scalar between 0 and 1. Detections
            %                          with scores less than the threshold
            %                          value are removed. Increase this value
            %                          to reduce false positives.
            %
            %                          Default: 0.5
            %
            % 'SelectStrongest'        A logical scalar. Set this to true to
            %                          eliminate overlapping object masks
            %                          based on their scores. This process is
            %                          often referred to as non-maximum
            %                          suppression. Set this to false if you
            %                          want to perform a custom selection
            %                          operation. When set to false, all the
            %                          segmented masks are returned.
            %
            %                          Default: true
            %
            % 'ExecutionEnvironment'   Specify what hardware resources will be used to
            %                          run the YOLO v8 detector. Valid values for
            %                          resource are:
            %
            %                          'auto' - Use a GPU if it is available, otherwise
            %                                   use the CPU.
            %
            %                          'gpu' - Use the GPU. To use a GPU, you must have
            %                                  Parallel Computing Toolbox(TM), and a
            %                                  CUDA-enabled NVIDIA GPU. If a suitable
            %                                  GPU is not available, an error message
            %                                  is issued.
            %
            %                          'cpu' - Use the CPU.
            %
            %                          Default: 'auto'
            %
            % The following name-value pair arguments control the writing of image
            % files. These arguments apply only when processing images in a datastore.
            %
            % 'MiniBatchSize'        A scalar to specify the size of the image batch
            %                        used to perform inference. This option can be used
            %                        to leverage batch inference to speed up processing,
            %                        comes at a cost of extra memory used. A higher value
            %                        of MiniBatchSize can result in out of memory errors,
            %                        depending on the hardware capabilities.
            %
            %                        Default: 1
            %
            % 'WriteLocation'        A scalar string or character vector to specify a
            %                        folder location to which extracted image files are
            %                        written. The specified folder must have
            %                        write permissions. If the folder already exists,
            %                        the next available number will be added as a suffix
            %                        to the folder name.
            %
            %                        Default: fullfile(pwd, 'SegmentObjectResults'), where
            %                        pwd is the current working directory.
            %
            % 'NamePrefix'           A scalar string or character vector to specify the
            %                        prefix applied to output image file names. For
            %                        input 2-D image inputs, the result MAT files
            %                        are named <prefix>_<imageName>.mat, where
            %                        imageName is the name of the input image
            %                        without its extension.
            %
            %                        Default: 'segmentObj'
            %
            % 'Verbose'              Set true to display progress information.
            %
            %                        Default: true
            %
            arguments
                obj
                im {validateImageInput}
                options.Threshold (1,1){mustBeNumeric, mustBePositive, mustBeLessThanOrEqual(options.Threshold, 1), mustBeReal} = 0.5
                options.SelectStrongest (1,1) logical = true
                options.ExecutionEnvironment {mustBeMember(options.ExecutionEnvironment,{'gpu','cpu','auto'})} = 'auto'
                options.Acceleration {mustBeMember(options.Acceleration,{'mex','none','auto'})} = 'auto'
                options.WriteLocation {mustBeTextScalar} = fullfile(pwd,'SegmentObjectResults')
                options.MiniBatchSize (1,1) {mustBeNumeric, mustBePositive, mustBeReal, mustBeInteger} = 1
                options.NamePrefix {mustBeTextScalar} = "segmentObj"
                options.Verbose (1,1) {validateLogicalFlag} = true
            end

            % Send the data to device or to host based on ExecutionEnvironment
            % option
            if(isequal(options.ExecutionEnvironment, 'auto'))
                if(canUseGPU)
                    options.ExecutionEnvironment = 'gpu';
                else
                    options.ExecutionEnvironment = 'cpu';
                end
            end

            % If writeLocation is set with a non-ds input, throw a warning
            if(~matlab.io.datastore.internal.shim.isDatastore(im) &&...
                    ~strcmp(options.WriteLocation, fullfile(pwd,'SegmentObjectResults')))

                warning(message('vision:solov2:WriteLocNotSupported'));
            end

            autoResize = true;
            castToGpuArray = ~isgpuarray(im);

            % Check if the input image is a single image or a batch
            if(matlab.io.datastore.internal.shim.isDatastore(im))
                nargoutchk(0,1);
                [varargout{1:nargout}] = ...
                    segmentObjectsInDatastore(obj, im,...
                    autoResize,...
                    options.MiniBatchSize,...
                    options,...
                    castToGpuArray);

            elseif(ndims(im)<=3)
                % Process Single image
                nargoutchk(0,4);
                miniBatchSize=1;
                [varargout{1:nargout}] =...
                    segmentObjectsInImgStack(obj, im,...
                    autoResize,...
                    miniBatchSize,...
                    options,...
                    castToGpuArray);
            elseif(ndims(im)==4)
                nargoutchk(0,4);
                [varargout{1:nargout}] = ...
                    segmentObjectsInImgStack(obj, im,...
                    autoResize,...
                    options.MiniBatchSize,...
                    options,...
                    castToGpuArray);
            else
                % Code flow shouldn't reach here (ensured by validation code).
                assert(false, 'Invalid image input.');
            end
        end
    end

    methods(Access=private)

        function [masks, labels, scores, bboxes] = segmentObjectsInImgStack(obj, im, autoResize, miniBatchSize, options, castToGpuArray)
            % This function dispatches batches for batch processing of
            % image Stacks.
            stackSize = size(im,4);

            masks = {};
            labels = {};
            scores = {};
            bboxes = {};

            % Process images from the imageStack, a minibatch at a time
            for startIdx = 1 : miniBatchSize : stackSize

                endIdx = min(startIdx+miniBatchSize-1, stackSize);

                imBatch = im(:,:,:,startIdx:endIdx);

                [masksCell, labelCell, scoreCell, boxCell] = ...
                    segmentObjectsInBatch(obj, imBatch,...
                    autoResize, options, castToGpuArray);

                masks = vertcat(masks, masksCell); %#ok<AGROW>
                labels = vertcat(labels, labelCell); %#ok<AGROW>
                scores = vertcat(scores, scoreCell); %#ok<AGROW>
                bboxes = vertcat(bboxes, boxCell); %#ok<AGROW>
            end

            % For a stack size = 1 (single image) output raw matrices
            % instead of cell arrays.
            if(isscalar(masks))
                masks = masks{1};
                % labels = labels{1};
                scores = scores{1};
                bboxes = bboxes{1};
            end
        end

        function outds = segmentObjectsInDatastore(obj, imds, autoResize, miniBatchSize, options, castToGpuArray)

            imdsCopy = copy(imds);
            imdsCopy.reset();

            % Get a new write location
            fileLocation = vision.internal.GetUniqueFolderName(options.WriteLocation);

            if(~exist(fileLocation, 'dir'))
                success = mkdir(fileLocation);
                if(~success)
                    throwAsCaller(MException('vision:solov2:folderCreationFailed',...
                        vision.getMessage('vision:solov2:folderCreationFailed')));
                end
            end

            % Handle verbose display
            printer = vision.internal.MessagePrinter.configure(options.Verbose);

            printer.linebreak();
            iPrintHeader(printer);
            msg = iPrintInitProgress(printer,'', 1);

            imIdx = 0;

            outFileList = [];
            % Process images from the datastore
            while (hasdata(imdsCopy))

                imBatch = [];
                fileNames = []; % Needed to build output names for result .matfiles

                % Build a minibatch worth of data
                for i = 1:miniBatchSize
                    if(~hasdata(imdsCopy))
                        break;
                    end
                    imIdx = imIdx + 1;
                    [img, imgInfo] = read(imdsCopy); %#ok<AGROW>

                    % Handle combineDS - use first cell, as the image is
                    % expected to be the first output.
                    if(iscell(imgInfo))
                        imgInfo = imgInfo{1};
                    end

                    %If the datastore doesn't expose filename, use the
                    % read index instead
                    if (isfield(imgInfo, 'Filename'))
                        [~,fileNames{i}] = fileparts(imgInfo.Filename); %#ok<AGROW>
                    else
                        fileNames{i} = num2str(imIdx); %#ok<AGROW>
                    end

                    if(iscell(img))
                        imBatch{i} = img{1}; % image should be the first output
                    else
                        imBatch{i} = img;
                    end
                end

                [masksCellSeg, labelCellSeg, scoreCellSeg, bboxCellSeg]  = ...
                    segmentObjectsInBatch(obj, imBatch, autoResize, options, castToGpuArray);

                if(iscell(masksCellSeg))
                    masksCell = masksCellSeg;
                    labelCell = labelCellSeg;
                    scoreCell = scoreCellSeg;
                    bboxCell = bboxCellSeg;
                else
                    masksCell{1} = masksCellSeg;
                    labelCell{1} = labelCellSeg;
                    scoreCell{1} = scoreCellSeg;
                    bboxCell{1} = bboxCellSeg;
                end

                % Write results to the disk
                for idx = 1:numel(masksCell)

                    matfilename = string(options.NamePrefix)+"_"+string(fileNames{idx})+".mat";

                    masks = masksCell{idx};
                    boxScore = scoreCell{idx};
                    boxLabel = labelCell{idx};
                    boxes = bboxCell{idx};

                    currFileName = fullfile(fileLocation, matfilename);

                    save(currFileName,...
                        "masks","boxScore","boxLabel","boxes");
                    outFileList = [outFileList; currFileName];
                end
                % Print number of processed images
                msg = iPrintProgress(printer, msg, imIdx+numel(masksCell)-1);
            end

            outds = fileDatastore(outFileList, 'FileExtensions', '.mat',...
                'ReadFcn', @(x)iSegmentObjectsReader(x));

            printer.linebreak(2);

        end


        function [masks, labels, scores, bboxes] = segmentObjectsInBatch(obj, im, autoResize, options, castToGpuArray)

            if(iscell(im))
                batchSize = numel(im);
                im = cat(4, im{:});
            else
                batchSize = size(im,4);
            end

            % Convert to gpuArray based on executionEnvironment.
            if castToGpuArray
                if (strcmp(options.ExecutionEnvironment,'auto') && canUseGPU) || strcmp(options.ExecutionEnvironment,'gpu')
                    im = gpuArray(im);
                end
            end

            % Preprocess input
            [im, info] = yolov8.preprocessInput(im, obj.InputSize(1:2),...
                obj.NormalizationStatistics.Mean, obj.NormalizationStatistics.StandardDeviation,...
                autoResize);
            im = dlarray(im, 'SSCB');

            % Predict on the yolov8 segmentation network
            fout = cell(size(obj.Network.OutputNames'));
            [fout{:}] = predict(obj.Network, im, "Acceleration", options.Acceleration);

            % Reshape predictions
            [detectionPriors, detectionDimension, maskPriors] = iReshapePredictions(fout);

            % Compute box and class priors
            [boxPriors, clsPriors] = iGetBoundingBoxesAndClasses(detectionPriors, detectionDimension);

            classes = obj.ClassNames;
            numClasses = size(classes,1);
            shape = size(im,1:2);

            % Obtain boxes, scores and labels
            if batchSize>1
                [bboxes, scores, labels, masks]  = yolov8.extractBatchDetections(fout{7,1}, boxPriors, clsPriors, maskPriors, classes, options.Threshold, batchSize, shape, info, castToGpuArray);
            else
                [bboxesNorm, scores, labelIds, fullDets]  = yolov8.extractDetections(boxPriors, clsPriors, maskPriors, numClasses, options.Threshold);
                masks = yolov8.extractMasks(fout{7,1}, fullDets(:,7:end), bboxesNorm, shape, info.OriginalSize);
                bboxesScaled = iScaleBoxes(shape, bboxesNorm, info.OriginalSize);
                bboxes = x1y1x2y2ToXYWH(bboxesScaled);
                % Convert classId to classNames.
                % Create categories of labels such that the order of the classes is retained.
                labels = categorical(classes,cellstr(classes));
                labels = labels(labelIds);
                if castToGpuArray
                    scores = gather(scores);
                    bboxes = gather(bboxes);
                    labels = gather(labels);
                end
            end
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

    methods(Static, Hidden)
        function [data, info] = preprocessInput(data, targetSize, ~, ~, ~)
            % Preprocess input data for inference
            if istable(data)
                data = table2cell(data);
            end

            info.OriginalSize = size(data);

            % Handle grayscale inputs
            if(size(data,3)==1)
                data = repmat(data,[1 1 3 1]);
            end

            Ibgr = flip(data,3);

            % Resize
            preprocesedImage = helper.preprocess(Ibgr,targetSize);
            data = flip(preprocesedImage,3);
        end

        function [bboxes, scores, labels, masks]  = extractBatchDetections(fout, boxPriors, clsPriors, maskPriors, classes, confThresh, batchSize, shape, info, castToGpuArray)
            numClasses = size(classes,1);
            bboxes = cell(batchSize, 1);
            scores = cell(batchSize, 1);
            labels = cell(batchSize, 1);
            masks = cell(batchSize, 1);
            for ii = 1:batchSize
                [bboxesNorm, scores{ii}, labelIds, fullDets] = yolov8.extractDetections(boxPriors(:,:,ii), clsPriors(:,:,ii), maskPriors(:,:,ii), numClasses, confThresh);
                masks{ii} = yolov8.extractMasks(fout(:,:,:,ii), fullDets(:,7:end), bboxesNorm, shape, info.OriginalSize);
                bboxesScaled = iScaleBoxes(shape, bboxesNorm, info.OriginalSize);
                bboxes{ii} = x1y1x2y2ToXYWH(bboxesScaled);
                % Convert classId to classNames.
                % Create categories of labels such that the order of the classes is retained.
                labelsMap = categorical(classes,cellstr(classes));
                labels{ii} = labelsMap(labelIds);
                if castToGpuArray
                    scores{ii} = gather(scores{ii});
                    bboxes{ii} = gather(bboxes{ii});
                    labels{ii} = gather(labels{ii});
                end
            end
        end

        function [bboxes, scores, labelIds, fullDets] = extractDetections(boxPriors, clsPriors, maskPriors, numClasses, confThresh)
            infOut = cat(2, boxPriors, clsPriors);
            pred = cat(2, infOut, maskPriors);
            maskIndex = 4 + numClasses;

            tmpVal = pred(:, 5:maskIndex);
            tmpMaxVal = max(tmpVal,[],2);
            boxCandidates = tmpMaxVal > confThresh;

            pred(:,1:4,:) = computeBoxes(pred(:,1:4));

            predFull = extractdata(pred(boxCandidates, :));

            box = predFull(:, 1:4);
            cls = predFull(:, 5:5 + numClasses-1);
            mask = predFull(:, 5 + numClasses:end);

            [clsConf,ind] = max(cls,[],2);
            fullDets = cat (2, box, clsConf, ind, mask);

            fullDets = fullDets(clsConf > confThresh,:);

            bboxesTmp = fullDets(:,1:4);
            scoresTmp = fullDets(:, 5);

            iou_thres = 0.8; % IoU threshold for NMS

            % Apply NMS
            [bboxes, scores, labelIds, idx] = selectStrongestBboxMulticlass(bboxesTmp, scoresTmp, ind, ...
                'RatioType', 'Min', 'OverlapThreshold', iou_thres);

            fullDets = fullDets(idx,:);
        end

        function [mask,downsampled_bboxes] = extractMasks(proto, masks_in, bboxes, shape, origShape)
            [mh, mw, c] = size(proto);
            [ih, iw] = deal(shape(1),shape(2));

            proto = extractdata(proto);
            protPermute = permute(proto,[3,2,1]);
            protoVal = reshape(protPermute,c,[]);

            maskTmp = masks_in*protoVal;

            % Match Python code
            maskTmpTrans = permute(maskTmp,[2,1]);
            masks = reshape(maskTmpTrans,mw,mh,[]);
            masks = permute(masks,[2,1,3]);

            % Vectorized bbox calculations
            scale = [mw./iw, mh./ih, mw./iw, mh./ih];
            downsampled_bboxes = bboxes .* scale;

            masks = iCropMasks(masks, downsampled_bboxes);

            % Resize masks efficiently
            mask = false([origShape(1:2), size(masks, 3)]);  % Preallocate as logical
            for i = 1:size(masks, 3)
                mask(:,:,i) = imresize(masks(:,:,i), [origShape(1), origShape(2)], 'bilinear') > 0;
            end
        end
    end
end

%--------------------------------------------------------------------------
function resultMasks = iCropMasks(masks, boxes)
[rows, cols, numBoxes] = size(masks);
[r, c] = ndgrid(1:rows, 1:cols);
resultMasks = zeros(size(masks), 'like', masks);  % Use same data type as input

% Vectorized box coordinates
boxes = boxes + 1;  % Add 1 to all coordinates at once
for i = 1:numBoxes
    logicalMask = (r >= boxes(i,2)) & (r < boxes(i,4)) & ...
        (c >= boxes(i,1)) & (c < boxes(i,3));
    resultMasks(:,:,i) = masks(:,:,i) .* logicalMask;
end
end

%--------------------------------------------------------------------------
function boxCentres = computeBoxes(boxCandidates)
dw = boxCandidates(:,3,:)./2;
dh = boxCandidates(:,4,:)./2;

% Initialize y with the same size as x
boxCentres = zeros(size(boxCandidates));

% Calculate top left x and y
boxCentres(:, 1, :) = boxCandidates(:, 1, :) - dw;
boxCentres(:, 2, :) = boxCandidates(:, 2, :) - dh;

% Calculate bottom right x and y
boxCentres(:, 3, :) = boxCandidates(:, 1, :) + dw;
boxCentres(:, 4, :) = boxCandidates(:, 2, :) + dh;

end

%--------------------------------------------------------------------------
function [detectionPriors, detectionDimension, maskPriors] = iReshapePredictions(fout)
% First three outputs correspond to mask priors
Z1Conv = fout{1,1};
batchSize = size(Z1Conv,4);
Z1Convmc = permute(Z1Conv,[2,1,3,4]);
Z1mc = reshape(Z1Convmc,[],32,batchSize);

Z2Conv = fout{2,1};
Z2Convmc = permute(Z2Conv,[2,1,3,4]);
Z2mc = reshape(Z2Convmc,[],32,batchSize);

Z3Conv = fout{3,1};
Z3Convmc = permute(Z3Conv,[2,1,3,4]);
Z3mc = reshape(Z3Convmc,[],32,batchSize);

maskPriors = cat(1, Z1mc, Z2mc, Z3mc);

% last 3 priors correspond to detection priors
Z1x = fout{4,1};
Z1ViewxCat = permute(Z1x,[2,1,3,4]);
detectionDimension{1,1} = size(Z1ViewxCat);
Z1xCat = reshape(Z1ViewxCat,[],144,batchSize);

Z2x = fout{5,1};
Z2ViewxCat = permute(Z2x,[2,1,3,4]);
detectionDimension{1,2} = size(Z2ViewxCat);
Z2xCat = reshape(Z2ViewxCat,[],144,batchSize);

Z3x = fout{6,1};
Z3ViewxCat = permute(Z3x,[2,1,3,4]);
detectionDimension{1,3} = size(Z3ViewxCat);
Z3xCat = reshape(Z3ViewxCat,[],144,batchSize);

detectionPriors = cat(1,Z1xCat,Z2xCat,Z3xCat);
end

%--------------------------------------------------------------------------
function [boxPriors, clsPriors] = iGetBoundingBoxesAndClasses(detectionPriors,detectionDimension)

stride = [8, 16, 32];
anchorMap{1,1} = zeros(detectionDimension{1,1});
anchorMap{2,1} = zeros(detectionDimension{1,2});
anchorMap{3,1} = zeros(detectionDimension{1,3});
anchorGrid = computeSegmentationAnchors(anchorMap, stride);
box = detectionPriors(:,1:64,:);
cls = detectionPriors(:,65:end,:);
boxPriors = zeros(size(detectionPriors,1),4,size(detectionPriors,3));

for i = 1:size(detectionPriors,3)
    % Decode boxes
    boxTmp = box(:,:,i);
    bboxData = reshape(boxTmp,[],16,4);

    % Apply softmax operation
    X = bboxData;
    X = X - max(X,[],2);
    X = exp(X);
    softmaxOut = X./sum(X,2);

    softmaxOut = permute(softmaxOut,[3,1,2]);
    softmaxOut = dlarray(single(softmaxOut),'SSCB');

    % Compute Distribution Focal Loss (DFL)
    weights = dlarray(single(reshape(0:15, [1, 1, 16])));
    bias = dlarray(single(0));

    convOut = dlconv(softmaxOut, weights, bias);
    convOut = extractdata(convOut);
    convOut = permute(convOut,[2,1]);

    % Transform distance (ltrb) to box (xywh)
    lt = convOut(:,1:2);
    rb = convOut(:,3:4);

    x1y1 = anchorGrid - lt;
    x2y2 = anchorGrid + rb;

    % Compute centre
    cxy = (x1y1 + x2y2)./2;

    % Compute width and height values
    wh = x2y2 - x1y1;

    % bbox values
    boxOut = cat(2,cxy,wh);

    % dbox values
    largestFeatureMapSize = detectionDimension{1,1}(1,1).*detectionDimension{1,1}(1,2);
    mulConst = [8.*ones(largestFeatureMapSize,1);16.*ones(largestFeatureMapSize./4,1);32.*ones(largestFeatureMapSize./16,1)];

    boxPriors(:,:,i) = boxOut.* mulConst;
end
clsPriors = sigmoid(dlarray(cls));
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
function network = iDownloadAndUpdatePretrainedModels(modelName, params)
data = downloadPretrainedYOLOv8 (modelName);
network = data.net;
if params.UpdateInputLayer
    network = iUpdateFirstConvChannelsAndInputLayer(network,params.InputSize);
end
end

%--------------------------------------------------------------------------
function model = downloadPretrainedYOLOv8(modelName)
% The downloadPretrainedYOLOv8 function downloads a YOLO v8 network
% pretrained on COCO dataset.

supportedNetworks = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"];
validatestring(modelName, supportedNetworks);

modelName = convertContainedStringsToChars(modelName);

netMatFileFullPath = fullfile(pwd, [modelName, 'Seg.mat']);

if ~exist(netMatFileFullPath,'file')
    fprintf(['Downloading pretrained ', modelName ,' network.\n']);
    fprintf('This can take several minutes to download...\n');
    url = ['https://github.com/matlab-deep-learning/Pretrained-YOLOv8-Network-For-Object-Detection/releases/download/v1.0.0/', [modelName,'Seg'], '.mat'];
    websave(netMatFileFullPath, url);
    fprintf('Done.\n\n');
else
    fprintf(['Pretrained ', modelName, ' network already exists.\n\n']);
end

model = load(netMatFileFullPath);
end

%--------------------------------------------------------------------------
function validateImageInput(in)

im = [];
if(isnumeric(in))
    im = in;
elseif(matlab.io.datastore.internal.shim.isDatastore(in))
    out = preview(in);
    if(iscell(out))
        if(isempty(out))
            im = [];
        else
            im = out{1};
        end
    else
        im = out;
    end
end

if(~validateImage(im)||isempty(im))
    throwAsCaller(MException('vision:solov2:invalidImageInput',...
        vision.getMessage('vision:solov2:invalidImageInput')));
end

end

%--------------------------------------------------------------------------
function tf = validateImage(in)
tf = isnumeric(in)&&...
    ndims(in)<=4 && ... && numdims should be less than 3
    (size(in,3)==3||size(in,3)==1); % gray scale or RGB image
end

%--------------------------------------------------------------------------
function validateLogicalFlag(in)
validateattributes(in,{'logical'}, {'scalar','finite', 'real'});
end

%--------------------------------------------------------------------------
function mustBeRGBSize(input)
% the size must either be [] or (1,3) with the channel dim=3

if(~isempty(input)&&length(input)>2)
    isValidChannelDim = length(input)==3 && input(3)==3;
else
    isValidChannelDim = false;
end

if~(isempty(input) || (isValidChannelDim))
    throwAsCaller(MException('vision:solov2:incorrectInputSize',...
        vision.getMessage('vision:solov2:incorrectInputSize')));
end
end

%--------------------------------------------------------------------------
function iCheckInputSize(inputSize)
validateattributes(inputSize, {'numeric'}, ...
    {'2d','nonempty','nonsparse',...
    'real','finite','integer','positive','nrows',1,'numel',3});

% if any(mod(inputSize(1:2),32))
%     error('Height and Width of input size [H W C] should be factor of 32');
% end

end

%--------------------------------------------------------------------------
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

%--------------------------------------------------------------------------
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

%--------------------------------------------------------------------------
function iValidateNormalizationStatsSize(statsSize,inputChannelSize)
if (numel(statsSize) == 2 && any(statsSize ~= [1 inputChannelSize])) || ...
        (numel(statsSize) == 3 && any(statsSize ~= [1 1 inputChannelSize])) || ...
        numel(statsSize) > 3
    error(message('visualinspection:yoloxObjectDetector:invalidNormalizationStatisticsSize',inputChannelSize));
end
end

%--------------------------------------------------------------------------
function x = iCreateDummyInput(inputSize)
dims = repmat('S',1,numel(inputSize)-1);
dims = [dims,'C'];
x = dlarray(zeros(inputSize),dims);
end

%--------------------------------------------------------------------------
function x = getExampleInputsFromNetwork(net)
x = getExampleInputs(net); % Populated when a user calls initialize on network without input layer.
if isempty(x)
    inputSize = net.Layers(1).InputSize;
    x = iCreateDummyInput(inputSize);
else
    x = iCreateDummyInput(size(x{1},[1 2 3]));
end
end

%--------------------------------------------------------------------------
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

%--------------------------------------------------------------------------
function newCoords = iScaleBoxes(img1_shape,coords,img0_shape)
% Rescale coords (xyxy) from img1_shape to img0_shape
gain = min(img1_shape(1) / img0_shape(1), img1_shape(2) / img0_shape(2));
pad = coder.nullcopy(zeros(1,2));
pad(1) = (img1_shape(2) - img0_shape(2) * gain)./ 2;
pad(2) = (img1_shape(1) - img0_shape(1) * gain)./ 2;

newCoords = coords - repmat(pad,size(coords,1),2);
newCoords = newCoords./gain;
newCoords(newCoords<0) = 0.1;

newCoords(:,1) = min(newCoords(:,1),img0_shape(2));
newCoords(:,2) = min(newCoords(:,2),img0_shape(1));
newCoords(:,3) = min(newCoords(:,3),img0_shape(2));
newCoords(:,4) = min(newCoords(:,4),img0_shape(1));

end

%--------------------------------------------------------------------------
function boxes = x1y1x2y2ToXYWH(boxes)
% Convert [x1 y1 x2 y2] boxes into [x y w h] format. Input and
% output boxes are in pixel coordinates. boxes is an M-by-4
% matrix.
boxes(:,3) = boxes(:,3) - boxes(:,1) + 1;
boxes(:,4) = boxes(:,4) - boxes(:,2) + 1;
end

%--------------------------------------------------------------------------
function anchorGrid = computeSegmentationAnchors(feats, strideValues)

gridCellOffset = 0.5;
n = 3;
anchorGridTmp = cell(n,1);
totalSize = 0;

for i = 1:n
    sz = size(feats{i});
    totalSize = totalSize + (sz(1).*sz(2));
    anchorGridTmp{i,1} = coder.nullcopy(zeros(sz(1).*sz(2),2));
end

for i=1:size(strideValues,2)
    [h,w,~,~]= size(feats{i});
    sx = (0:h-1)+gridCellOffset;
    sy = (0:w-1)+gridCellOffset;
    [sy,sx]= meshgrid(sy,sx);
    anchorGridTmp{i,1} = cat(2, sx(:), sy(:));
end
anchorGrid = cat(1,anchorGridTmp{:});
end

%--------------------------------------------------------------------------
function iPrintHeader(printer)
printer.print('Running YOLO v8 network');
printer.linebreak();
printer.print('-----------------------');
printer.linebreak();
end

%--------------------------------------------------------------------------
function nextMessage = iPrintInitProgress(printer, prevMessage, k)
nextMessage = getString(message('vision:solov2:verboseProgressTxt',k));
updateMessage(printer, prevMessage(1:end-1), nextMessage);
end

%--------------------------------------------------------------------------
function updateMessage(printer, prevMessage, nextMessage)
backspace = sprintf(repmat('\b',1,numel(prevMessage))); % figure how much to delete
printer.print([backspace nextMessage]);
end

%--------------------------------------------------------------------------
function nextMessage = iPrintProgress(printer, prevMessage, k)
nextMessage = getString(message('vision:solov2:verboseProgressTxt',k));
updateMessage(printer, prevMessage, nextMessage);
end

%-----------------------------------------------------------------------
function out = iSegmentObjectsReader(filename)
% SegmentObjectsReader is a custom mat file reader for results of segmentObjects()
% stored as mat-files. This custom reader when used with file datastore returns
% a 1x3 or, 1x4 cell array containing the masks, labels, scores (and boxes) (as generated
% by segmentObjects(detector, ds))

data = load(filename);

out{1} = data.masks;
out{2} = data.boxLabel;
out{3} = data.boxScore;
out{4} = data.boxes;

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
function tf = iIsValidDataType(value)
    tf = iscategorical(value) || iscellstr(value) || isstring(value);
end

%--------------------------------------------------------------------------
function tf = iHasDuplicates(value)
    tf = ~isequal(value, unique(value, 'stable'));
end
