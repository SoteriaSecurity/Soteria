#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "YoloOnnxModel.h"


std::queue<cv::Mat> frameQueue;
std::mutex queueMutex;
std::condition_variable queueCV;
bool done = false; // To signal the end of video processing

void captureFrames(cv::VideoCapture& cap) {
    while (true) {
        cv::Mat frame;
        {
            // Lock the queue
            std::lock_guard<std::mutex> lock(queueMutex);

            if (!cap.read(frame)) {
                std::cout << "End of video or failed to read frame?" << std::endl;
                done = true;
                queueCV.notify_all();
                break;
            }

            frameQueue.push(frame);
        }
        queueCV.notify_one();
    }
}

void processFrames(YoloOnnxModel& yolo) {
    while (true) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCV.wait(lock, [] { return !frameQueue.empty() || done; });

            if (frameQueue.empty() && done) break;

            frame = frameQueue.front();
            frameQueue.pop();
        }

        auto detections = yolo.infer(frame, 0.5f, 0.4f);

        for (const auto& [box, label] : detections) {
            cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 2);
            cv::putText(frame, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
        }

        cv::imshow("Video Feed", frame);

        if (cv::waitKey(1) == 'q') {
            done = true;
            break;
        }
    }
}


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

    std::thread captureThread(captureFrames, std::ref(cap));
    std::thread processThread(processFrames, std::ref(yolo));

    // Join threads
    captureThread.join();
    processThread.join();

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
