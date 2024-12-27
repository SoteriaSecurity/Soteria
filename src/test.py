from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("../include/model/yolov8s.pt")
    model.export(format="onnx")
