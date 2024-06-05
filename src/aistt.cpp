//
// Created by gopizza on 2024-06-04.
//
#include "aistt.h"
#include <iostream>
#include "../common/logger.h"
#include "../common/common.h"

void AISTT::Initialize() {
    std::cout << "AISTT::Initialize()" << std::endl;
}

void AISTT::Release() {
    std::cout << "AISTT::Release()" << std::endl;
    engine_->Release();
    buffer_->release();
    if(use_cam_) camera_->Release();
    preprocess_->release();
    postprocess_->release();
    stream_.reset();

    input_dims_.reset();
    output_dims_.reset();
}

bool AISTT::LoadEngine()
{
    std::cout << "AISTT::LoadEngine()" << std::endl;
    gotrt::ModelParams model_params;
    model_params.wight_path = "weights/best.onnx";
    model_params.input_tensor_names.push_back("images");
    model_params.output_tensor_names.push_back("output0");

    engine_->LoadEngine(model_params, input_dims_, output_dims_);
    return true;
}

bool AISTT::CreateModule(int _org_height, int _org_width, bool _use_cam, bool _is_obb)
{
    std::cout << "AISTT::CreateModule()" << std::endl;
    org_image_height_ = _org_height;
    org_image_width_ = _org_width;
    use_cam_ = _use_cam;
    is_obb_ = _is_obb;

    engine_ = std::make_shared<gotrt::GoEngine>();
    engine_->Initialize();
    LoadEngine();
    buffer_ = std::make_shared<gotrt::GoBuffer>();
    buffer_->initialize(1, input_dims_, output_dims_);
    preprocess_ = std::make_shared<PreProcess>();
    preprocess_->initialize(input_dims_, output_dims_, org_image_height_, org_image_width_);
    if(_is_obb) postprocess_ = std::make_shared<PostProcessOBB>();
    else postprocess_ = std::make_shared<PostProcessBase>();
    postprocess_->initialize(input_dims_, output_dims_);

    if(!_use_cam) return true;
    camera_ = std::make_shared<DAIStream>();
    camera_->Initialize(org_image_height_, org_image_width_);
    return true;
}

//void AISTT::Inference()
//{
//    std::cout << "AISTT::Inference()" << std::endl;
//    buffer_->cpyInputToDevice();
//    engine_->Inference(buffer_);
//    buffer_->cpyOutputToHost();
//}
//
//void AISTT::Preprocess()
//{
//    std::cout << "AISTT::Preprocess()" << std::endl;
//    if(use_cam_) camera_->get_frame(buffer_->input_tensor.cpu_tensor);
//    preprocess_->preprocess(buffer_);
//}