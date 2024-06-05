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
    if(camera_!=nullptr) camera_->Release();
    preprocess_->release();
    postprocess_->release();
    stream_.reset();

    input_dim_.reset();
    output_dims_.reset();
}

bool AISTT::LoadEngine()
{
    std::cout << "AISTT::LoadEngine()" << std::endl;
    gotrt::ModelParams model_params;
    model_params.wight_path = "weights/best.onnx";
    model_params.input_tensor_names.push_back("images");
    model_params.output_tensor_names.push_back("output0");

    engine_->LoadEngine(model_params, input_dim_, output_dims_);
    input_tensor_size_ = sizeof(float);
    for(int i = 0; i < input_dim_->nbDims; i++) input_tensor_size_ *= input_dim_->d[i];
    output_tensor_size_.resize(output_dims_->size());
    for(int i = 0; i < output_dims_->size(); i++)
    {
        output_tensor_size_[i] = sizeof(float);
        for(int j = 0; j < (*output_dims_)[i].nbDims; j++) output_tensor_size_[i] *= (*output_dims_)[i].d[j];
    }
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
    buffer_->initialize(1, input_dim_, output_dims_);
    preprocess_ = std::make_shared<PreProcess>();
    preprocess_->initialize(input_dim_, output_dims_, org_image_height_, org_image_width_);
    if(_is_obb)
    {
        postprocess_ = std::make_shared<PostProcessOBB>();
        boxes.resize(100, std::vector<float>(10, 0));
    }
    else
    {
        postprocess_ = std::make_shared<PostProcessBase>();
        boxes.resize(100, std::vector<float>(7, 0));
    }
    postprocess_->initialize(input_dim_, output_dims_);

    std::string file_name = "test.jpg";
    if(_use_cam) camera_ = std::make_shared<DAIStream>();
    else if (ends_with(file_name, ".jpg") || ends_with(file_name, ".png")) camera_ = std::make_shared<ImageStream>();
    else if (ends_with(file_name, ".mp4") || ends_with(file_name, ".avi")) camera_ = std::make_shared<VideoStream>();
    else {
        camera_ = nullptr;
        return false;
    }
    camera_->Initialize(file_name, org_image_height_, org_image_width_);

    return true;
}

void AISTT::Inference()
{
    std::cout << "AISTT::Inference()" << std::endl;
    buffer_->cpyInputToDevice();
    engine_->Inference(buffer_->getDeviceBindings().data());
    buffer_->cpyOutputToHost(0);
}

void AISTT::Preprocess()
{
    std::cout << "AISTT::Preprocess()" << std::endl;
    if(camera_==nullptr)
        return;
    cv::Mat image = camera_->GetFrame().clone();
    preprocess_->opencv_preprocess(&image);
    memcpy(buffer_->getInputTensor(), preprocess_->opencv_preprocess(&image), input_tensor_size_);
}

void AISTT::Postprocess()
{
    std::cout << "AISTT::Postprocess()" << std::endl;
    float* output = buffer_->getOutputTensor(0);
    postprocess_->clear();
    postprocess_->normalize(output, 0.25f);
    auto normalize_box= postprocess_->weighted_boxes_fusion();
    postprocess_->denormalize(normalize_box, 100, &boxes);
}

bool AISTT::Draw()
{
    std::cout << "AISTT::Draw()" << std::endl;
    cv::Mat image = preprocess_->resize_image_->clone();
    for(int i = 0; i < boxes.size(); i++)
    {
        cv::rectangle(image, cv::Point(boxes[i][0], boxes[i][1]), cv::Point(boxes[i][2], boxes[i][3]), cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("result", image);
    if(cv::waitKey(1) == 'q')
        return false;
    return true;
}