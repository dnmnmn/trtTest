cmake_minimum_required(VERSION 3.28)

project(wintrt CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# include / link
if(WIN32)
    # libfoundation
    include_directories("../libFoundation")
    link_directories("../Library")
    # cuda
    include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")
    link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64")
    # tensorRT
    include_directories("C:/Program Files/TensorRT/include")
    include_directories("common/windows")
    link_directories("C:/Program Files/TensorRT/lib")
    # opencv
    include_directories("C:/library/opencv/build/include")
    link_directories("../Library/opencv")
    # depthai
    include_directories("C:/library/depthai/include")
    include_directories("C:/library/depthai/include/depthai-shared/3rdparty")
    include_directories("C:/library/depthai/lib/cmake/depthai/dependencies/include")
    link_directories("C:/library/depthai/lib")
    link_directories("../Library/depthai")
endif()

add_executable(wintrt main.cpp
        src/model/utils/postprocess.h
        src/model/utils/postprocess.cpp
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
        src/stream/input_stream.cpp
        src/stream/input_stream.h
        src/model/utils/preprocess.cpp
        src/model/utils/preprocess.h
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
        src/model/detection/yolov8_det.cpp
        src/model/detection/yolov8_det.h
        src/model/segmentation/yolov8_seg.cpp
        src/model/segmentation/yolov8_seg.h
        src/model/utils/CUDA_preprocess.cu
        src/model/utils/postprocess_seg.cpp
        src/model/utils/postprocess_seg.h
        src/model/utils/postprocess_det.cpp
        src/model/utils/postprocess_det.h
        src/model/utils/CUDA_postprocess.cu
        src/data/data_store.cpp
        src/data/data_store.h
        src/model/model.cpp
        src/model/model.h
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries( wintrt
            PRIVATE libFounDation
            PRIVATE cuda cudnn cublas cudart nppig
            PRIVATE nvinfer nvonnxparser nvparsers nvinfer_plugin zlibwapi
            PRIVATE opencv_world490d
            PRIVATE depthai-cored depthai-opencvd
        )
else()
    target_link_libraries( wintrt
            PRIVATE libFounDation
            PRIVATE cuda cudnn cublas cudart nppig
            PRIVATE nvinfer nvonnxparser nvparsers nvinfer_plugin zlibwapi
            PRIVATE opencv_world490
            PRIVATE depthai-core depthai-opencv
    )
endif()

