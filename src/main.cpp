#include <filesystem>
#include <iostream>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

void preprocess(const cv::Mat& frame, std::vector<float>& input_tensor, const int64_t input_width, const int64_t input_height) {
    cv::Mat resized, blob;
    cv::resize(frame, resized, cv::Size(static_cast<int>(input_width), static_cast<int>(input_height)));
    resized.convertTo(blob, CV_32F, 1 / 255.0);

    // Flatten the image and rearrange channels
    std::vector<cv::Mat> channels(3);
    cv::split(blob, channels);
    for (const auto& channel : channels) {
        input_tensor.insert(input_tensor.end(), channel.begin<float>(), channel.end<float>());
    }
}


std::vector<std::string> loadClassNames(const std::string& file_path) {
    std::vector<std::string> class_names;
    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_path << std::endl;
        return class_names;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            class_names.push_back(line);
        }
    }

    file.close();
    return class_names;
}

int main() {
    const int columnsPerOutput = 84;
    auto DARK_BLUE = cv::Scalar(255, 0, 0);

    // initialize onnx

    // std::cout << "ONNX Runtime version: " << Ort::GetVersionString() << std::endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
    std::wstring model_path = std::filesystem::absolute("include/yolov8n.onnx").wstring();
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int64_t input_width = input_shape[2];
    int64_t input_height = input_shape[3];
    // size_t input_tensor_size = input_width * input_height * 3;

    // initialize camera
    cv::VideoCapture cap(0, cv::CAP_DSHOW);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera." << std::endl;
        return -1;
    }

    cv::namedWindow("Live Camera Feed", cv::WINDOW_AUTOSIZE);
    cv::Mat frame;

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Empty frame captured." << std::endl;

            // reinit cam (hope and pray)
            cap.release();
            cap.open(0, cv::CAP_V4L2);
            if (!cap.isOpened()) {
                std::cerr << "Failed to reinitialize capture." << std::endl;
                break;
            }
        }

        // stores resized frames as channels in input_tensor
        std::vector<float> input_tensor;
        preprocess(frame, input_tensor, input_width, input_height);

        Ort::AllocatorWithDefaultOptions allocator;

        // input -> output tensors
        std::array<int64_t, 4> input_shape_arr = {1, 3, input_height, input_width}; // 1 - single frame, 3 - RGB

        Ort::Value input_tensor_onnx = Ort::Value::CreateTensor<float>(
            allocator.GetInfo(),
            input_tensor.data(),
            input_tensor.size(),
            input_shape_arr.data(),
            input_shape_arr.size()
        );

        std::vector<std::string> input_names;
        std::vector<std::string> output_names;

        for (size_t i = 0; i < session.GetInputCount(); ++i) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            if (!input_name) {
                throw std::runtime_error("Failed to retrieve input name.");
            }
            input_names.emplace_back(input_name.get());
        }
        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            if (!output_name) {
                throw std::runtime_error("Failed to retrieve output name.");
            }
            output_names.emplace_back(output_name.get());
        }

        std::vector<const char*> input_names_cstr(input_names.size());
        std::vector<const char*> output_names_cstr(output_names.size());

        for (size_t i = 0; i < input_names.size(); ++i) {
            input_names_cstr[i] = input_names[i].c_str();
        }
        for (size_t i = 0; i < output_names.size(); ++i) {
            output_names_cstr[i] = output_names[i].c_str();
        }

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names_cstr.data(),
            &input_tensor_onnx,
            1,
            output_names_cstr.data(),
            1
        );

        // process the new detections
        const auto* output_data = output_tensors.front().GetTensorData<float>();
        size_t num_detections = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

        if (num_detections % columnsPerOutput != 0) {
            std::cerr << "Error: Output tensor dimensions mismatch." << std::endl;
            return -1;
        }

        std::vector<std::tuple<cv::Rect, int, float>> detections;

        std::vector<std::string> class_names = loadClassNames(std::filesystem::absolute("include/coco.names").string());
        for (size_t i = 0; i < num_detections / columnsPerOutput; ++i) {
            const float* prediction = output_data + i * columnsPerOutput;
            float confidence = prediction[4];
            if (confidence >= 70) {
                int w = static_cast<int>(prediction[2]);
                int h = static_cast<int>(prediction[3]);
                int x = static_cast<int>(prediction[0] - w / 2.0f);
                int y = static_cast<int>(prediction[1] - h / 2.0f);

                if (int classId = static_cast<int>(prediction[5]); classId >= 0 && classId < class_names.size()) {
                    std::cout << class_names[classId] << ", " << confidence << std::endl;
                    std::string label = class_names[classId] + " (" + std::to_string(static_cast<int>(confidence)) + ")";

                    cv::Rect box(x, y, w, h);
                    cv::rectangle(frame, box, DARK_BLUE, 2);
                    cv::putText(frame, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                        0.5, DARK_BLUE, 1);
                }
            }
        }

        cv::imshow("Live Camera Feed", frame);

        if (cv::waitKey(5) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}