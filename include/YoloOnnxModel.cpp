#include "YoloOnnxModel.h"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <filesystem>

YoloOnnxModel::YoloOnnxModel(const std::string& model_path, const std::string& class_names_path, bool isGpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "YOLOv8"),
      session_(env_, std::filesystem::absolute(model_path).wstring().c_str(), Ort::SessionOptions()) {
    // Load class names
    class_names_ = loadClassNames(class_names_path);

    // Configure session options
    Ort::SessionOptions session_options;
    if (isGpu) {
        auto cuda_available = std::find(Ort::GetAvailableProviders().begin(), Ort::GetAvailableProviders().end(),
                                        "CUDAExecutionProvider");
        if (cuda_available != Ort::GetAvailableProviders().end()) {
            OrtCUDAProviderOptions cuda_options;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }
        else {
            std::cout << "CUDA not available. Falling back to CPU." << std::endl;
        }
    }

    // Retrieve input shape
    auto input_shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    input_width_ = input_shape[2];
    input_height_ = input_shape[3];

    // Retrieve input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session_.GetInputCount(); ++i) {
        input_names_.emplace_back(session_.GetInputNameAllocated(i, allocator).release());
    }
    for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
        output_names_.emplace_back(session_.GetOutputNameAllocated(i, allocator).release());
    }
}

std::vector<std::string> YoloOnnxModel::loadClassNames(const std::string& file_path) {
    std::vector<std::string> class_names;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error opening class names file: " << file_path << std::endl;
        return class_names;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            class_names.push_back(line);
        }
    }
    return class_names;
}

void YoloOnnxModel::preprocess(const cv::Mat& frame, std::vector<float>& input_tensor) const {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_width_, input_height_));
    resized.convertTo(resized, CV_32F, 1 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (int c = 0; c < 3; ++c) {
        input_tensor.insert(input_tensor.end(), channels[c].begin<float>(), channels[c].end<float>());
    }
}

std::vector<std::tuple<cv::Rect, std::string>> YoloOnnxModel::infer(const cv::Mat& frame, float conf_threshold,
                                                                    float nms_threshold) {
    std::vector<float> input_tensor;
    preprocess(frame, input_tensor);
    std::array<int64_t, 4> input_shape = {1, 3, input_height_, input_width_};
    Ort::Value input_tensor_onnx = Ort::Value::CreateTensor<float>(
        Ort::AllocatorWithDefaultOptions().GetInfo(),
        input_tensor.data(),
        input_tensor.size(),
        input_shape.data(),
        input_shape.size()
    );
    auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor_onnx, 1,
                                       output_names_.data(), output_names_.size());
    return postprocess(frame.cols, frame.rows, output_tensors, conf_threshold, nms_threshold);
}

std::vector<std::tuple<cv::Rect, std::string>> YoloOnnxModel::postprocess(
    const int64_t& image_width,
    const int64_t& image_height,
    const std::vector<Ort::Value>& outputTensors,
    float confThreshold,
    float nmsThreshold) {
    std::vector<std::tuple<cv::Rect, std::string>> detections;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<std::tuple<cv::Rect, int, float>> oldBoxes; // to prevent overlap

    const auto* outputData = outputTensors.front().GetTensorData<float>();
    const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<std::string> classNames =
        loadClassNames(std::filesystem::absolute("include/coco.names").string());
    const size_t num_classes = outputShape[1] - 4;
    const size_t num_detections = outputShape[2];

    if (num_detections == 0) {
        return detections;
    }

    for (size_t i = 0; i < num_detections; ++i) {
        int classId = 0;
        float confidence = 0;

        // max class confidence => class ID
        for (int c = 0; c < num_classes; ++c) {
            const float classConf = outputData[(4 + c) * num_detections + i];
            if (classConf > confidence) {
                confidence = classConf;
                classId = c;
            }
        }

        if (confidence >= confThreshold) {
            const int h = static_cast<int>(outputData[3 * num_detections + i]);
            const int w = static_cast<int>(outputData[2 * num_detections + i]);
            const int x = static_cast<int>(outputData[0 * num_detections + i] - w / 2.0);
            const int y = static_cast<int>(outputData[1 * num_detections + i] - h / 2.0);

            boxes.emplace_back(cv::Rect(x, y, w, h));
            confidences.push_back(confidence);
            class_ids.push_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (const int idx : indices) {
        const cv::Rect& box = boxes[idx];
        const int classId = class_ids[idx];
        std::string label = classNames[classId] + " (" + std::to_string(static_cast<int>(confidences[idx] * 100)) + ")";
        detections.emplace_back(box, label);
    }

    return detections;
}
