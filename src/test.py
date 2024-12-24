from ultralytics import YOLO
import onnx

if __name__ == "__main__":
    model_path = "../include/yolov8n.onnx"
    onnx_model = onnx.load(model_path)

    # Inspect the output tensor
    output_info = onnx_model.graph.output
    for output in output_info:
        print(f"Output Name: {output.name}")
        print(f"Shape: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")

    model = YOLO("../include/yolov8n.onnx")  # Replace with your model
    results = model("image.jpg")  # Replace with your input image

    print("1 1 1 1 1")
    print(results)

    print("2 2 2 2 2")
    # Access detections (assuming batch size = 1)
    detections = results[0].boxes
    print(detections)
    print("*******")
    # data, confidence, label
    print(detections.data)
    print(detections.conf)
    print(detections.cls)
    print(detections.xywh)

    print("3 3 3 3 3")
    detection_array = detections.xyxy.numpy()

    print(detection_array)

    print("4 4 4 4 4")
    print(dir(detections))

    # Check the docstring for more details
    help(detections)
