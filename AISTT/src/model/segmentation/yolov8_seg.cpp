//
// Created by gopizza on 2024-06-10.
//

#include "yolov8_seg.h"
#include "../../../common/logger.h"
#include "Json/JsonObject.h"
#include "FileSystem/FileSystem.h"

void Yolov8Seg::Initialize() {
    std::cout << "Yolov8Seg Initialize" << std::endl;
    JsonObject config_json;
    string config_path = FileSystem::get_app_path() + "//config.json";
    config_json.load(config_path);
    org_image_width_ = config_json.get_int("GoEngine/Resolution/Width");
    org_image_height_ = config_json.get_int("GoEngine/Resolution/Height");
    input_index = config_json.get_int("GoEngine/Detect/InputIdx");
    auto output_idx = config_json.get_array("GoEngine/Detect/SegOutputIdx");
    for(int i = 0; i < output_idx.size(); i++) output_index_vector.push_back((int)round(output_idx[i]));

    // Engine Initialization
    engine_ = std::make_shared<gotrt::GoEngine>();
    engine_->Initialize();
    input_dim_ = std::make_shared<nvinfer1::Dims32>();
    output_dims_ = std::make_shared<std::vector<nvinfer1::Dims32>>();

    std::string segment_weight = config_json.get_string("GoEngine/Path/SegmentWeights");
    LoadEngine(segment_weight);
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
    postprocess_ = std::make_shared<PostProcessSeg>();
    boxes.resize(100, std::vector<float>(38, 0));
    postprocess_->initialize(input_dim_, output_dims_);
    buffer_ = std::make_shared<gotrt::GoBuffer>();
    buffer_->initialize(1, input_dim_, output_dims_, org_image_height_, org_image_width_, 104);
    engine_->SetBuffer(buffer_);
}

//void Yolov8Seg::Release() {
//    std::cout << "Yolov8Seg Release" << std::endl;
//    preprocess_->release();
//    postprocess_->release();
//    input_dim_.reset();
//    output_dims_.reset();
//    engine_->Release();
//}

//void Yolov8Seg::Inference() {
//    timer_.start();
//    buffer_->cpyInputToDevice();
//    engine_->Inference();
//    buffer_->cpyOutputToHost(output_index_vector[0]);
//    sample::gLogInfo << "Inference time: " << timer_.end() << "ms" << std::endl;
//}

void Yolov8Seg::PreProcessFunc(cv::Mat *image) {
    //timer_.start();
    cv::Mat img;
    image->convertTo(img, CV_32FC3);
    buffer_->cpyOrginInputToDevice(&img);
    auto f = preprocess_->opencv_preprocess(image);
    memcpy(engine_->GetInputData(), f, input_tensor_size_);
    //sample::gLogInfo << "Preprocess time: " << timer_.end() << "ms" << std::endl;
}

void Yolov8Seg::PostProcessFunc() {
    //timer_.start();
    // float* output = engine_->GetOutputTensor(output_index_vector[0]);
    auto g_tensor = engine_->GetOutputTensor(output_index_vector[0]);

    postprocess_->ConvertGpuClassScoreTensor(g_tensor, &(buffer_->class_score_tensor), buffer_->stream, 4, 104, 0.35f);

    std::cout << "class_score_tensor size: " << (buffer_->class_score_tensor).tensor_size << std::endl;
    //Model::SaveText((buffer_->class_score_tensor).cpu_tensor, (buffer_->class_score_tensor).tensor_size, "class_score_tensor.txt");
    //Model::SaveText(g_tensor->cpu_tensor, g_tensor->tensor_size, "org_output_tensor.txt");
    postprocess_->ConvertGpuInverseTensor(&(buffer_->class_score_tensor), &(buffer_->inverse_tensor), buffer_->stream);
    //Model::SaveText((buffer_->inverse_tensor).cpu_tensor, (buffer_->inverse_tensor).tensor_size, "inverse_tensor.txt");
    postprocess_->clear();
    postprocess_->normalize((buffer_->inverse_tensor).cpu_tensor, 0.25f);
    std::vector<std::vector<float>> normalize_box = postprocess_->nms();
    postprocess_->denormalize(normalize_box, &boxes);
    //sample::gLogInfo << "Postprocess time: " << timer_.end() << "ms" << std::endl;
}

bool Yolov8Seg::Draw()
{
    // cv::Mat image = preprocess_->resize_image_->clone();
    cv::Mat image = data_store_.lock()->org_image_->clone();
    //cv::Mat image;
    //cv::resize(image, org_image, cv::Size(input_dim_->d[3], input_dim_->d[2]));

    for(int i = 0; i < boxes.size(); i++)
    {
        cv::rectangle(image, cv::Point(boxes[i][0], boxes[i][1]), cv::Point(boxes[i][2], boxes[i][3]), cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("result", image);
    auto key = cv::waitKey(1);
    if(key == 'q' || key == 'Q'){
        cv::destroyAllWindows();
        return false;
    }
    return true;
}

//void Yolov8Seg::Debug()
//{
//    postprocess_->debug(&(buffer_->class_score_tensor));
//}

bool Yolov8Seg::LoadEngine(std::string _model_path) {
    std::cout << "Yolov8Seg::LoadEngine()" << std::endl;
    gotrt::ModelParams model_params;
    model_params.wight_path = _model_path;
    model_params.input_index = input_index;
    model_params.output_index_vector.push_back(output_index_vector[0]);
    engine_->LoadEngine(model_params, input_dim_, output_dims_);

    return true;
}