//
// Created by gopizza on 2024-06-04.
//

#ifndef WINTRT_AISTT_H
#define WINTRT_AISTT_H
#pragma once
#include <string>
#include <memory>
#include "opencv2/opencv.hpp"
#include "NvInfer.h"

#include "trt/Gobuffer.h"
#include "trt/go_engine.h"
#include "preprocess.h"
#include "postprocess.h"
#include "stream/go_camera.h"
#include "stream/dai_stream.h"

class AISTT {
public:
    AISTT() {};
    ~AISTT() {};
    void Initialize();
    void Release();

    bool CreateModule(int _org_height=1080, int _org_width=1920, bool _use_cam=false, bool _is_obb=false);
    bool LoadEngine();
    void Inference();
    void Preprocess();
    void Postprocess();
    void Draw();

private:
    std::shared_ptr<gotrt::GoEngine> engine_;
    std::shared_ptr<gotrt::GoBuffer> buffer_;
    std::shared_ptr<PreProcess> preprocess_;
    std::shared_ptr<PostProcess> postprocess_;
    std::shared_ptr<InputStream> camera_;
    std::unique_ptr<cudaStream_t> stream_;

    std::shared_ptr<nvinfer1::Dims32> input_dims_;
    std::shared_ptr<std::vector<nvinfer1::Dims32>> output_dims_;

private:
    int org_image_height_;
    int org_image_width_;
    bool use_cam_;
    bool is_obb_;
};

#endif //WINTRT_GOENGINE_H
