from ultralytics import YOLO
import yaml


class yolov8TrainerClass():
    def __init__(self, baseModel, imageSize):
        self.epochs = 1;
        self.imgSize = imageSize;
        self.model = YOLO(baseModel)

    def trainYOLOv8(self, config, epochs):
        results = self.model.train(data= config, epochs=epochs, imgsz=self.imgSize)
        return results
    
    def exportModel(self, onnxPath):
        model = YOLO(onnxPath)
        model.export(format='onnx',opset=14)


    def getClassNames(self,fileName):
        with open(fileName) as stream:
            try:
                dictValue = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return dictValue
