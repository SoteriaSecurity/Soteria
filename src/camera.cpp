#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "YoloOnnxModel.h"

int main() {
    std::string pathToModel = "include/model/yolo11n.onnx";
    std::string pathToNames = "include/coco.names";
    constexpr bool isGPU = false; // set to false if your GPU does not support CUDA 😭😭😭

    YoloOnnxModel yolo(pathToModel, pathToNames, isGPU);

    // initialize camera
    cv::VideoCapture cap(0, cv::CAP_DSHOW);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera." << std::endl;
        return -1;
    }

    cv::namedWindow("Live Camera Feed", cv::WINDOW_NORMAL);
    cv::Mat frame;

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Empty frame captured." << std::endl;
        }

        auto detections = yolo.infer(frame, 0.5f, 0.4f);

        for (const auto& [box, label] : detections) {
            cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 2);
            cv::putText(frame, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
        }

        cv::imshow("Live Camera Feed", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
