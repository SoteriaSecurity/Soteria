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

std::vector<std::tuple<cv::Rect, std::string>> postprocess(
    const int64_t &image_width,
    const int64_t &image_height,
    const std::vector<Ort::Value> &outputTensors,
    const float confThreshold
    ) {
    std::vector<std::tuple<cv::Rect, std::string>> detections;
    std::vector<std::tuple<cv::Rect, int, float>> oldBoxes; // to prevent overlap

    const auto* outputData = outputTensors.front().GetTensorData<float>();
    const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<std::string> classNames = loadClassNames(std::filesystem::absolute("include/coco.names").string());
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
            const int x = static_cast<int>(outputData[0 * num_detections + i]  - w / 2.0);
            const int y = static_cast<int>(outputData[1 * num_detections + i]  - h / 2.0);

            cv::Rect box(x, y, w, h);
            bool alrChecked = false;

            for (auto &old : oldBoxes) {
                cv::Rect oldBox = std::get<0>(old);
                const int oldId = std::get<1>(old);
                const float oldConf = std::get<2>(old);

                // swap old and new box if the new box should be taking priority
                // (has a higher confidence than old)
                if (oldConf < confidence) {
                    const auto &temp = oldBox;
                    oldBox = box;
                    box = temp;
                }

                const cv::Rect overlap = oldBox & box;
                // if the entire thing is overlap
                // or a lot of it overlaps and its the same object
                if (overlap.area() >= box.area() * 0.95 ||
                    oldId == classId && overlap.area() >= 0.8 * box.area()) {
                    alrChecked = true;
                    break;
                }
            }

            if (!alrChecked) {
                std::string label = classNames[classId] + " (" + std::to_string(confidence) + ")";

                oldBoxes.emplace_back(box, classId, confidence);
                detections.emplace_back(box, label);
            }
        }
    }

    return detections;
}

int main() {
    auto DARK_BLUE = cv::Scalar(255, 0, 0);

    // initialize onnx
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
    std::wstring model_path = std::filesystem::absolute("include/yolov8n.onnx").wstring();
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int64_t input_width = input_shape[2];
    int64_t input_height = input_shape[3];

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

        auto outputTensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names_cstr.data(),
            &input_tensor_onnx,
            1,
            output_names_cstr.data(),
            1
        );

        const std::vector<std::tuple<cv::Rect, std::string>> detections = postprocess(
            input_width,
            input_height,
            outputTensors,
            0.6f
            );

        for (const auto &detection : detections) {
            const cv::Rect& box = std::get<0>(detection);
            const std::string& label = std::get<1>(detection);

            cv::rectangle(frame, box, DARK_BLUE, 2);
            cv::putText(frame, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                0.5, DARK_BLUE, 1);
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