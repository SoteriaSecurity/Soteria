set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_minimum_required(VERSION 3.10)

project(Soteria)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(ONNX_RUNTIME_DIR "C:/tools/onnxruntime") # change to your system?
set(ONNX_RUNTIME_INCLUDE_DIR "${ONNX_RUNTIME_DIR}/include")
set(ONNX_RUNTIME_LIB_DIR "${ONNX_RUNTIME_DIR}/lib")
include_directories(${ONNX_RUNTIME_INCLUDE_DIR})

add_executable(camera
        src/camera.cpp)
add_executable(video
        src/video.cpp)
target_link_libraries(camera ${OpenCV_LIBS})
target_link_libraries(video ${OpenCV_LIBS})

if(WIN32)
    target_link_libraries(camera "${ONNX_RUNTIME_LIB_DIR}/onnxruntime.lib")
    target_link_libraries(video "${ONNX_RUNTIME_LIB_DIR}/onnxruntime.lib")
else()
    target_link_libraries(camera "${ONNX_RUNTIME_LIB_DIR}/libonnxruntime.so")
    target_link_libraries(video "${ONNX_RUNTIME_LIB_DIR}/libonnxruntime.so")
endif()

# Output build instructions
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "ONNX Runtime include directory: ${ONNX_RUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library directory: ${ONNX_RUNTIME_LIB_DIR}")
