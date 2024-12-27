#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <filesystem>
#include <string>


int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestONNX");
    std::cout << "ONNX Runtime version: " << Ort::GetVersionString() << std::endl;
    Ort::SessionOptions session_options;

    std::wstring model_path = std::filesystem::absolute("include/yolov8s.onnx").wstring();
    std::wcout << L"Loading model from: " << model_path << std::endl;
    try {
        Ort::Session session(env, model_path.c_str(), session_options);
        std::cout << "ONNX Runtime initialized successfully and model loaded!" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error initializing ONNX Runtime: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}