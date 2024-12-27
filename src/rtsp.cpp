#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "YoloOnnxModel.h"

std::queue<cv::Mat> frameQueue;
std::mutex queueMutex;
std::condition_variable condVar;
bool stopThreads = false;

void captureFrames(cv::VideoCapture& cap) {
    cv::Mat frame;
    while (!stopThreads) {
        cap >> frame;
        if (frame.empty()) continue;

        std::lock_guard<std::mutex> lock(queueMutex);
        if (frameQueue.size() < 10) {
            frameQueue.push(frame);
            condVar.notify_one();
        }
    }
}

void processFrames(YoloOnnxModel& yolo) {
    while (!stopThreads) {
        std::unique_lock<std::mutex> lock(queueMutex);
        condVar.wait(lock, [] { return !frameQueue.empty() || stopThreads; });

        if (!frameQueue.empty()) {
            cv::Mat frame = frameQueue.front();
            frameQueue.pop();
            lock.unlock();

            auto detections = yolo.infer(frame, 0.4f, 0.4f);
            for (const auto& [box, label] : detections) {
                cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 2);
                cv::putText(frame, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
            }

            cv::imshow("Live Camera Feed (RTSP)", frame);
            if (cv::waitKey(1) == 'q') {
                stopThreads = true;
            }
        }
    }
}

int main() {
    std::string pathToModel = "include/model/yolo11n.onnx";
    std::string pathToNames = "include/coco.names";
    constexpr bool isGPU = false; // set to false if your GPU does not support CUDA 😭😭😭

    YoloOnnxModel yolo(pathToModel, pathToNames, isGPU);
    cv::VideoCapture cap("rtsp://localhost:8554/stream", cv::CAP_FFMPEG); // make filename args in future
    cap.set(cv::CAP_PROP_BUFFERSIZE, 3);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera." << std::endl;
        return -1;
    }

    std::thread captureThread(captureFrames, std::ref(cap));
    std::thread processThread(processFrames, std::ref(yolo));

    captureThread.join();
    processThread.join();

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
