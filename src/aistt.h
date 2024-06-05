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
#include "stream/image_stream.h"
#include "stream/video_stream.h"


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
    bool Draw();



private:
    std::shared_ptr<gotrt::GoEngine> engine_;
    std::shared_ptr<gotrt::GoBuffer> buffer_;
    std::shared_ptr<PreProcess> preprocess_;
    std::shared_ptr<PostProcess> postprocess_;
    std::shared_ptr<InputStream> camera_= nullptr;
    std::unique_ptr<cudaStream_t> stream_;

    std::shared_ptr<nvinfer1::Dims32> input_dim_;
    std::shared_ptr<std::vector<nvinfer1::Dims32>> output_dims_;
    cv::Mat image_;

private:
    int org_image_height_;
    int org_image_width_;
    bool use_cam_;
    bool is_obb_;
    int32_t input_tensor_size_;
    std::vector<int32_t> output_tensor_size_;
    std::vector<std::vector<float>> boxes;

private:
    inline bool ends_with(std::string const & value, std::string const & ending)
    {
        if (ending.size() > value.size()) return false;
        return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
    }
};

#endif //WINTRT_GOENGINE_H
