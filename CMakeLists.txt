cmake_minimum_required(VERSION 3.28)

project(wintrt)

set(CMAKE_CXX_STANDARD 17)

# include / link
#if(WIN32)
    # cuda
    include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")
    link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64")
    # tensorRT
    include_directories("C:/Program Files/TensorRT/include")
    include_directories("common/windows")
    link_directories("C:/Program Files/TensorRT/lib")
    # opencv
    include_directories("C:/library/opencv/build/include")
    link_directories("C:/library/opencv/build/x64/vc16/lib")
    # depthai
    include_directories("C:/library/depthai/include")
    include_directories("C:/library/depthai/include/depthai-shared/3rdparty")
    include_directories("C:/library/depthai/lib/cmake/depthai/dependencies/include")
    link_directories("C:/library/depthai/lib")
#endif()

add_executable(wintrt main.cpp
        src/postprocess.h
        src/postprocess.cpp
        "common/argsParser.h"
        "common/BatchStream.h"
        "common/buffers.h"
        "common/common.h"
        "common/EntropyCalibrator.h"
        "common/ErrorRecorder.h"
        "common/getOptions.cpp"
        "common/getOptions.h"
        "common/half.h"
        "common/logger.h"
        "common/logger.cpp"
        "common/logging.h"
        "common/parserOnnxConfig.h"
        "common/safeCommon.h"
        "common/sampleConfig.h"
        "common/sampleDevice.h"
        "common/sampleEngines.h"
        "common/sampleEngines.cpp"
        "common/sampleInference.h"
        "common/sampleInference.cpp"
        "common/sampleOptions.h"
        "common/sampleOptions.cpp"
        "common/sampleReporting.h"
        "common/sampleReporting.cpp"
        "common/sampleUtils.h"
        "common/sampleUtils.cpp"
        "common/windows/getopt.h"
        "common/windows/getopt.c"
        src/stream/go_camera.cpp
        src/stream/go_camera.h
        src/Timer.cpp
        src/Timer.h
        src/preprocess.cpp
        src/preprocess.h
        src/trt/GoBuffer.cpp
        src/trt/Gobuffer.h
        src/aistt.cpp
        src/aistt.h
        src/trt/go_logger.cpp
        src/trt/go_logger.h
        src/trt/go_engine.cpp
        src/trt/go_engine.h
        src/stream/video_stream.cpp
        src/stream/video_stream.h
        src/stream/camera_stream.cpp
        src/stream/camera_stream.h
        src/stream/image_stream.cpp
        src/stream/image_stream.h
        src/stream/dai_stream.cpp
        src/stream/dai_stream.h
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries( wintrt
            PRIVATE cudnn cublas cudart
            PRIVATE nvinfer nvonnxparser nvparsers nvinfer_plugin zlibwapi
            PRIVATE opencv_world490d
            PRIVATE depthai-cored depthai-opencvd
        )
else()
    target_link_libraries( wintrt
            PRIVATE cudnn cublas cudart
            PRIVATE nvinfer nvonnxparser nvparsers nvinfer_plugin zlibwapi
            PRIVATE opencv_world490
            PRIVATE depthai-core depthai-opencv
    )
endif()