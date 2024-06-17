//
// Created by gopizza on 2024-06-07.
//

#include "yolov8_det.h"
#include "../../../common/logger.h"
#include "FileSystem/FileSystem.h"
#include "Json/JsonObject.h"

void Yolov8Det::Initialize() {
    std::cout << "yolo_v8_det::initialize()" << std::endl;
    JsonObject config_json;
    string config_path = FileSystem::get_app_path() + "//config.json";
    config_json.load(config_path);
    org_image_width_ = config_json.get_int("GoEngine/Resolution/Width");
    org_image_height_ = config_json.get_int("GoEngine/Resolution/Height");
    use_obb_ = (bool)config_json.get_int("GoEngine/Utils/UseOBB");

    input_index = config_json.get_int("GoEngine/Detect/InputIdx");
    if(use_obb_)
    {
        auto output_idx = config_json.get_array("GoEngine/Detect/OBBOutputIdx");
        for(int i = 0; i < output_idx.size(); i++) output_index_vector.push_back((int)round(output_idx[i]));
    }
    else output_index_vector.push_back(config_json.get_int("GoEngine/Detect/OutputIdx"));

    // Engine Initialization
    engine_ = std::make_shared<gotrt::GoEngine>();
    engine_->Initialize();
    input_dim_ = std::make_shared<nvinfer1::Dims32>();
    output_dims_ = std::make_shared<std::vector<nvinfer1::Dims32>>();

    std::string detect_weight = "";
    if(use_obb_)
        detect_weight = config_json.get_string("GoEngine/Path/DetectOBBWeights");
    else
        detect_weight = config_json.get_string("GoEngine/Path/DetectWeights");
    LoadEngine(detect_weight);
    input_tensor_size_ = sizeof(float);
    for(int i = 0; i < input_dim_->nbDims; i++) input_tensor_size_ *= input_dim_->d[i];
    output_tensor_size_.resize(output_dims_->size());
    for(int i = 0; i < output_dims_->size(); i++)
    {
        output_tensor_size_[i] = sizeof(float);
        for(int j = 0; j < (*output_dims_)[i].nbDims; j++) output_tensor_size_[i] *= (*output_dims_)[i].d[j];
    }

    // PreProcess Initialization
    preprocess_ = std::make_shared<PreProcess>();
    preprocess_->initialize(input_dim_, output_dims_, org_image_height_, org_image_width_);

    // PostProcess Initialization
    if(use_obb_)
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
    buffer_ = std::make_shared<gotrt::GoBuffer>();
    buffer_->initialize(1, input_dim_, output_dims_, org_image_height_, org_image_width_);
    engine_->SetBuffer(buffer_);
}

//void Yolov8Det::Release() {
//    std::cout << "yolo_v8_det::release()" << std::endl;
//    preprocess_->release();
//    postprocess_->release();
//    input_dim_.reset();
//    output_dims_.reset();
//    engine_->Release();
//}

//void Yolov8Det::Inference() {
//    timer_.start();
//    buffer_->cpyInputToDevice();
//    engine_->Inference();
//    buffer_->cpyOutputToHost(output_index_vector[0]);
//    sample::gLogInfo << "Inference time: " << timer_.end() << "ms" << std::endl;
//}

void Yolov8Det::PreProcessFunc(cv::Mat* image) {
    //std::cout << "yolo_v8_det::preprocess()" << std::endl;
    timer_.start();
    memcpy(engine_->GetInputData(), preprocess_->opencv_preprocess(image), input_tensor_size_);
    //sample::gLogInfo << "Preprocess time: " << timer_.end() << "ms" << std::endl;
}

void Yolov8Det::PostProcessFunc() {
    //std::cout << "yolo_v8_det::postprocess()" << std::endl;
    timer_.start();
    float* output = engine_->GetOutputData(output_index_vector[0]);
    postprocess_->clear();
    postprocess_->normalize(output, 0.25f);
    std::vector<std::vector<float>> normalize_box;
    if(use_obb_)
        normalize_box= postprocess_->weighted_boxes_fusion();
    else
        normalize_box = postprocess_->nms();
    postprocess_->denormalize(normalize_box, &boxes);
    //sample::gLogInfo << "Postprocess time: " << timer_.end() << "ms" << std::endl;
}

bool Yolov8Det::Draw()
{
    // cv::Mat image = preprocess_->resize_image_->clone();
    cv::Mat image(input_dim_->d[3], input_dim_->d[2], CV_8UC3);
    cv::Mat org_image = data_store_.lock()->org_image_->clone();
    cv::resize(org_image, image, cv::Size(input_dim_->d[3], input_dim_->d[2]));

    if(use_obb_)
    {
        for(int i = 0; i < boxes.size(); i++)
        {
            cv::Point2f vertices[4];
            vertices[0] = cv::Point(boxes[i][0], boxes[i][1]);
            vertices[1] = cv::Point(boxes[i][2], boxes[i][3]);
            vertices[2] = cv::Point(boxes[i][4], boxes[i][5]);
            vertices[3] = cv::Point(boxes[i][6], boxes[i][7]);
            for(int j = 0; j < 4; j++)
            {
                cv::line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
        }
    }
    else{
        for(int i = 0; i < boxes.size(); i++)
        {
            cv::rectangle(image, cv::Point(boxes[i][0], boxes[i][1]), cv::Point(boxes[i][2], boxes[i][3]), cv::Scalar(0, 255, 0), 2);
        }
    }
    cv::imshow("result", image);
    auto key = cv::waitKey(2);
    if(key == 'q' || key == 'Q'){
        cv::destroyAllWindows();
        return false;
    }
    return true;
}

//void Yolov8Det::Debug()
//{
//    // float* output = engine_->GetOutputTensor(output_index_vector[0]);
//    postprocess_->debug(buffer_->getGpuOutputTensor(output_index_vector[0]));
//}

bool Yolov8Det::LoadEngine(std::string _model_path) {
    std::cout << "yolo_v8_det::LoadEngine()" << std::endl;
    gotrt::ModelParams model_params;
    model_params.wight_path = _model_path;
    model_params.input_index = input_index;
    model_params.output_index_vector.push_back(output_index_vector[0]);
    engine_->LoadEngine(model_params, input_dim_, output_dims_);

    return true;
}