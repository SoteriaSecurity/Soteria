#include <filesystem>
#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include "YoloOnnxModel.h"

int main() {
    std::string pathToModel = "include/model/yolo11n.onnx";
    std::string pathToNames = "include/coco.names";
    constexpr bool isGPU = false; // set to false if your GPU does not support CUDA 😭😭😭

    YoloOnnxModel yolo(pathToModel, pathToNames, isGPU);

    std::string video_path = std::filesystem::absolute("include/video1.mp4").string(); // Specify your video file path here
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file." << std::endl;
        return -1;
    }

    cv::namedWindow("Video Feed", cv::WINDOW_AUTOSIZE);
    cv::Mat frame;

    constexpr int EXPECTED_MS_PER_FRAME = 100; // ~70 for 8n, ~135 for 8s (no gpu)
    int frame_skip = static_cast<int>(std::round(EXPECTED_MS_PER_FRAME / cap.get(cv::CAP_PROP_FPS)));
    if (frame_skip < 1) frame_skip = 1; // Ensure we skip at least one frame

    int frame_count = 0;

    auto videoStart = std::chrono::high_resolution_clock::now();

    while (true) {
        // auto start = std::chrono::high_resolution_clock::now();
        if (!cap.read(frame)) { // Read next frame from video
            std::cout << "Failed to read frame (end of video?)" << std::endl;
            break;
        }

        // frame_count++;
        // // Skip frames
        // if (frame_count % frame_skip != 0) {
        //     continue;
        // }

        auto detections = yolo.infer(frame, 0.5f, 0.4f);

        for (const auto& [box, label] : detections) {
            cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 2);
            cv::putText(frame, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
        }

        cv::imshow("Video Feed", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
        // auto end = std::chrono::high_resolution_clock::now();
        // std::cout << "frame - " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    }

    auto videoEnd = std::chrono::high_resolution_clock::now();
    std::cout << "video - " << std::chrono::duration_cast<std::chrono::milliseconds>(videoEnd - videoStart).count() << " ms";

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
