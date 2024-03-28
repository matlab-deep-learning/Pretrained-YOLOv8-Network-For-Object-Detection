# Pretrained YOLO v8 Network For Object Detection

This repository provides multiple pretrained YOLO v8[1] object detection networks for MATLAB®, trained on the COCO 2017[2] dataset. These object detectors can detect 80 different object categories including [person, car, traffic light, etc](/src/%2Bhelper/getCOCOClasess.m).

**Creator**: MathWorks Development

**Includes Codegen support**: ✔  

**Includes transfer learning script**: ❌  

## License
The software and model weights are released under the [GNU Affero General Public License v3.0](LICENSE). For alternative licensing, contact [Ultralytics Licensing](https://www.ultralytics.com/license).

## Requirements
- MATLAB® R2023b or later
- Computer Vision Toolbox™
- Deep Learning Toolbox™
- Deep Learning Toolbox Converter for ONNX Model Format
- (optional) MATLAB® Coder for code generation
- (optional) GPU Coder for code generation

## Getting Started
Download or clone this repository to your machine and open it in MATLAB®.

### Setup
Add path to the models directory.

```matlab
addpath('models');
```
### Download the pretrained network
Use the code below to download the pretrained network.

```matlab
% Load YOLO v8 model
modelName = 'yolov8s';
model = helper.downloadPretrainedYOLOv8(modelName);
net = model.yolov8Net;
```

modelName of the pretrained YOLO v8 deep learning model, specified as one of these:
- yolov8n
- yolov8s
- yolov8m
- yolov8l
- yolov8x

Following is the description of various YOLO v8 models available in this repo:

| Model         |                                      Description                                                                                                                   |
|-------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| yolov8n       |   Nano pretrained YOLO v8 model optimized for speed and efficiency.                                                                                                |
| yolov8s       |   Small pretrained YOLO v8 model balances speed and accuracy, suitable for applications requiring real-time performance with good detection quality.               |
| yolov8m       |   Medium pretrained YOLO v8 model offers higher accuracy with moderate computational demands.                                                                      |
| yolov8l       |   Large pretrained YOLO v8 model prioritizes maximum detection accuracy for high-end systems, at the cost of computational intensity.                              |
| yolov8x       |   Extra Large YOLOv8 model is the most accurate but requires significant computational resources, ideal for high-end systems prioritizing detection performance.   |

### Detect Objects Using Pretrained YOLO v8
To perform object detection on an example image using the pretrained model, utilize the provided code below. Alternatively, you can execute the `runInference.m` script for the same purpose.

```matlab
% Read test image.
I = imread(fullfile('data','inputTeam.jpg'));

% Get classnames for COCO dataset.
classNames = helper.getCOCOClassNames;
numClasses = size(classNames,1);

% Load YOLO v8 small network.
modelName = 'yolov8s';
data = helper.downloadPretrainedYOLOv8(modelName);
det = data.yolov8Net;

% Perform detection using pretrained model.
executionEnvironment = 'auto';
[bboxes, scores, labelIds] = detectYOLOv8(det, I, numClasses, executionEnvironment);

% Map labelIds back to labels.
labels = classNames(labelIds);

% Visualize detection results.
annotations = string(labels) + ': ' + string(scores);
Iout = insertObjectAnnotation(I, 'rectangle', bboxes, annotations);
figure, imshow(Iout);
```
![Results](/data/resultsTeam.png)


## Metrics and Evaluation

### Size and Accuracy Metrics

| Model         | Input image resolution | Size (MB) | mAP  |
|-------------- |:----------------------:|:---------:|:----:|
| yolov8n       |       640 x 640        |  10.7     | 37.3 |
| yolov8s       |       640 x 640        |  37.2     | 44.9 |
| yolov8m       |       640 x 640        |  85.4     | 50.2 |
| yolov8l       |       640 x 640        |  143.3    | 52.9 |
| yolov8x       |       640 x 640        |  222.7    | 53.9 |

mAP for models trained on the COCO dataset is computed as average over IoU of .5:.95.

## Deployment
Code generation enables you to generate code and deploy YOLO v8 on multiple embedded platforms. The list of supported platforms is shown below:

| Target                             |  Support  |   Notes                     |
|------------------------------------|:---------:|:---------------------------:|
| GPU Coder                          |     ✔     |    run `gpuCodegenYOLOv8.m` |
| MATLAB Coder                       |     ✔     |    run `codegenYOLOv8.m`    |

To deploy YOLO v8 to GPU Coder, run `gpuCodegenYOLOv8.m`. This script calls the `yolov8Predict.m` entry point function and generate CUDA code for it. It will run the generated MEX and give an output.
For more information about codegen, see [Deep Learning with GPU Coder](https://in.mathworks.com/help/gpucoder/gpucoder-deep-learning.html).

## Network Overview
YOLO v8 is one of the best performing object detectors and is considered as an improvement to the existing YOLO variants such as YOLO v5, and YOLOX.

Following are the key features of the YOLO v8 object detector compared to its predecessors:
- Improved Accuracy: YOLO v8 is expected to offer enhanced accuracy in object detection compared to its previous versions. This improvement can lead to more precise and reliable detection results.
- Better Speed and Efficiency: YOLO v8 may have optimizations that allow it to achieve faster processing speeds while maintaining high accuracy. This can be crucial for real-time applications or scenarios with limited computational resources.
- Advanced Backbone Network: YOLO v8 might incorporate a more advanced backbone network architecture, such as Darknet-53 or a similar architecture, which can enable better feature extraction and representation.
- Enhanced Object Classification: YOLO v8 may introduce improvements in object classification capabilities, allowing for more accurate and detailed classification of detected objects. 


## References
[1] https://github.com/ultralytics/ultralytics

[2] Lin, T., et al. "Microsoft COCO: Common objects in context. arXiv 2014." arXiv preprint arXiv:1405.0312 (2014).


Copyright 2024 The MathWorks, Inc.
