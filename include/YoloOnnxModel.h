#ifndef YOLO_ONNX_MODEL_H
#define YOLO_ONNX_MODEL_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <tuple>

class YoloOnnxModel {
public:
    YoloOnnxModel(const std::string& model_path, const std::string& class_names_path, bool isGpu = false);
    ~YoloOnnxModel() = default;

    void preprocess(const cv::Mat& frame, std::vector<float>& input_tensor) const;
    std::vector<std::tuple<cv::Rect, std::string>> infer(const cv::Mat& frame, float conf_threshold, float nms_threshold);

private:
    Ort::Env env_;
    Ort::Session session_;
    std::vector<std::string> class_names_;
    int64_t input_width_;
    int64_t input_height_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    static std::vector<std::string> loadClassNames(const std::string& file_path);
    std::vector<std::tuple<cv::Rect, std::string>> postprocess(const int64_t& image_width, const int64_t& image_height, const std::vector<Ort::Value>& output_tensors, float conf_threshold, float nms_threshold);
};

#endif // YOLO_ONNX_MODEL_H
