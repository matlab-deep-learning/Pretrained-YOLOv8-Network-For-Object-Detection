# Pretrained YOLO v8 Network For Object Detection

This repository provides multiple pretrained YOLO v8[1] object detection networks for MATLAB®, trained on the COCO 2017[2] dataset. These object detectors can detect 80 different object categories including [person, car, traffic light, etc](/src/%2Bhelper/getCOCOClasess.m). [![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=matlab-deep-learning/Pretrained-YOLOv8-Network-For-Object-Detection)

**Creator**: MathWorks Development

**Includes Codegen support**: ✔  

**Includes transfer learning script**: ✔  

**Includes Simulink support script**: ✔  

## License
The software and model weights are released under the [GNU Affero General Public License v3.0](LICENSE). For alternative licensing, contact [Ultralytics Licensing](https://www.ultralytics.com/license).

## Requirements
- MATLAB® R2023b or later
- Computer Vision Toolbox™
- Deep Learning Toolbox™
- Deep Learning Toolbox Converter for ONNX Model Format
- (optional) Visual Studio C++ compiler for training on Windows
- (optional) MATLAB® Coder for code generation
- (optional) GPU Coder for code generation

## Getting Started
Download or clone this repository to your machine and open it in MATLAB®.

### Download the pretrained network
Use the code below to download the pretrained network.

```matlab
% Load YOLO v8 model
det = yolov8ObjectDetector('yolov8s');

% Analyze loaded model
analyzeNetwork(det.Network);
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
To perform object detection on an example image using the pretrained model, utilize the provided code below.

```matlab
% Read test image.
I = imread(fullfile('data','inputTeam.jpg'));

% Load YOLO v8 small network.
det = yolov8ObjectDetector('yolov8s');

% Perform detection using pretrained model.
[bboxes, scores, labels] = detect(det, I);

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

## Transfer learning
Training of a YOLO v8 object detector requires GPU drivers. Additionally, when training on Windows, the Visual Studio C++ compiler is required to facilitate the process.

### Setup
Run `installUltralytics.m` to install the required Python files and set up the Python environment for training YOLOv8. This MATLAB script automates downloading and setting up a standalone Python environment tailored for YOLOv8 training. It determines the system architecture, downloads the appropriate Python build, extracts it, and configures MATLAB settings to use this Python interpreter. Finally, it installs the Ultralytics package and its dependencies using `pip`.

### Obtain data
Use the code below to download the multiclass dataset, or follow the subsequent steps to create a custom dataset.
```matlab
helper.downloadMultiClassData()
```

#### Dataset format for custom dataset
Labels for training YOLO v8 must be in YOLO format, with each image having its own *.txt file. If an image contains no objects, a *.txt file is not needed. Each *.txt file should have one row per object in the format: class xCenter yCenter width height, where class numbers start from 0, following a zero-indexed system. Box coordinates should be in normalized xywh format, ranging from 0 to 1. If the coordinates are in pixels, divide xCenter and width by the image width, and yCenter and height by the image height as shown below.

<img src="/data/datasetFormat.png" alt="datasetFormat" width="500"/>

The label file corresponding to the above image contains 3 chairs (class 2).

![labelFormat](/data/s1_96.png)

Refer to the code below to convert bounding box [xCord, yCord, wdt, ht] to [xCenter, yCenter, wdt, ht] format.
```matlab
boxes(:,1) = boxes(:,1) + (boxes(:,3)./2);
boxes(:,2) = boxes(:,2) + (boxes(:,4)./2);
```

Refer to the code below to normalize ground truth bounding box [xCenter, yCenter, wdt, ht] with respect to input image.
```matlab
% Read input image.
[Irow, Icol, ~] = size(inpImage);

% Normalize xCenter and width of ground truth bounding box.
xCenter = xCenter./Icol;
wdt = wdt./Icol;

% Normalize yCenter and height of ground truth bounding box.
yCenter = yCenter./Irow;
ht = ht./Irow;
```

#### Dataset directory structure for custom dataset
When using custom dataset for YOLO training, organize training and validation images and labels as shown in the datasets example directory below. It is mandatory to have both training and validation data to train YOLO v8 network.

<img src="/data/datasetDir.png" alt="datasetDir" width="500"/>

### Transfer learning
Run below code to train YOLO v8 object detector on multiclass object detection dataset. For more information about dataset and evaluation, see [Multiclass Object Detection Using YOLO v2 Deep Learning](https://in.mathworks.com/help/vision/ug/multiclass-object-detection-using-yolo-v2-deep-learning.html) example.

```matlab
yolov8Det = trainYOLOv8ObjectDetector('data.yaml','yolov8n.pt', ImageSize=[720 720 3], MaxEpochs=10);
```

### Infer trained model
```matlab
% Detect Multiple Indoor Objects
I = imread(fullfile('data','indoorTest.jpg'));
[bbox,score,label] = detect(yolov8Det,I);

annotatedImage = insertObjectAnnotation(I,"rectangle",bbox,label,LineWidth=4,FontSize=24);
figure
imshow(annotatedImage)
```

## Deployment
Code generation enables you to generate code and deploy YOLO v8 on multiple embedded platforms. The list of supported platforms is shown below:

| Target                             |  Support  |   Notes                     |
|------------------------------------|:---------:|:---------------------------:|
| GPU Coder                          |     ✔     |    run `gpuCodegenYOLOv8.m` |
| MATLAB Coder                       |     ✔     |    run `codegenYOLOv8.m`    |

To deploy YOLO v8 to GPU Coder, run `gpuCodegenYOLOv8.m`. This script calls the `yolov8Predict.m` entry point function and generate CUDA code for it. It will run the generated MEX and give an output.
For more information about codegen, see [Deep Learning with GPU Coder](https://in.mathworks.com/help/gpucoder/gpucoder-deep-learning.html).

## Simulink
Simulink is a block diagram environment used to design systems with multidomain models, simulate before moving to hardware, and deploy without writing code. For more information about simulink, see [Get Started with Simulink](https://in.mathworks.com/help/simulink/getting-started-with-simulink.html)

```matlab
% Read test image.
I = imread(fullfile('data','inputTeam.jpg'));

% Open Simulink model.
open('yolov8SimulinkSupport.slx')
```
To run the simulation, click `Run` from the `Simulation` tab.

The output will be logged to the workspace variable `out` from the Simulink model.

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
