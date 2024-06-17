//
// Created by gopizza on 2024-06-04.
//
#include "aistt.h"
#include <iostream>
#include "../../common/logger.h"
#include <FileSystem/FileSystem.h>
#include <Json/JsonObject.h>

void AISTT::Initialize() {
    std::cout << "AISTT::Initialize()" << std::endl;
    string config_path = FileSystem::get_app_path() + "//config.json";
    cout << config_path << endl;

    JsonObject config;
    if (FileSystem::exist(config_path) == false)
    {
        // Create the config file
        JsonObject config_json;
        // path
        config_json.set_string("GoEngine/Path/LogDir", (string)"//log");
        config_json.set_string("GoEngine/Path/DetectWeights", (string)"weights//best.onnx");
        config_json.set_string("GoEngine/Path/DetectOBBWeights", (string)"weights//best_obb.onnx");
        config_json.set_string("GoEngine/Path/SegmentWeights", (string)"weights//best_seg.onnx");
        config_json.set_string("GoEngine/Path/Stream", (string)"C://data//video//full//35.avi");
        // threads
        config_json.set_int("GoEngine/Threads/NumDetect", 1);
        config_json.set_int("GoEngine/Threads/NumSegment", 3);
        // interval
        config_json.set_int("GoEngine/Interval/Detect", 2);
        config_json.set_int("GoEngine/Interval/Segment", 1);
        // Resolution
        config_json.set_int("GoEngine/Resolution/Width", 4032);
        config_json.set_int("GoEngine/Resolution/Height", 1504);
        // Detect
        config_json.set_int("GoEngine/Detect/InputIdx", 0);
        config_json.set_int("GoEngine/Detect/OutputIdx", 0);
        std::vector<int> output_idx = {4, 3, 2, 1, 0};
        config_json.set_array("GoEngine/Detect/OBBOutputIdx", output_idx);
        // Segment
        config_json.set_int("GoEngine/Segment/InputIdx", 0);
        config_json.set_int("GoEngine/Segment/OutputIdx", 0);
        // Utils
        config_json.set_int("GoEngine/Utils/UseCamera", 0);
        config_json.set_int("GoEngine/Utils/UseOBB", 0);
        // Log Level
        config_json.set_int("GoEngine/LogLevel", 0);
        // Save the config file
        config_json.save(config_path);
    }

    // Load the config file
    JsonObject config_json;
    config_json.load(config_path);
    bool use_cam = (bool)config_json.get_int("GoEngine/Utils/UseCamera");
    std::string file_name = config_json.get_string("GoEngine/Path/Stream");
    int org_image_height = config_json.get_int("GoEngine/Resolution/Height");
    int org_image_width = config_json.get_int("GoEngine/Resolution/Width");
    auto segment_only = (bool)config_json.get_int("GoEngine/Utils/SegmentOnly");
    if(FileSystem::exist(file_name) == false)
    {
        sample::gLogInfo << "Error: AISTT::Initialize() - Invalid file path" << std::endl;
        assert(false);
        return;
    }

    // Initialize the data store
    data_store_ = std::make_shared<DataStore>();
    data_store_->Initialize();

    // Initialize the camera
    if(use_cam) camera_ = std::make_shared<DAIStream>();
    else if (FileSystem::ends_with(file_name, ".jpg") || FileSystem::ends_with(file_name, ".png")) camera_ = std::make_shared<ImageStream>();
    else if (FileSystem::ends_with(file_name, ".mp4") || FileSystem::ends_with(file_name, ".avi")) camera_ = std::make_shared<VideoStream>();
    else {
        camera_ = nullptr;
        sample::gLogInfo << "Error: AISTT::Initialize() - Invalid file format" << std::endl;
        return;
    }
    camera_->Initialize(file_name, org_image_height, org_image_width);
    camera_->SetDataStore(data_store_);

    // Initialize the model
    if(segment_only)
    {
        seg_model_ = std::make_shared<Yolov8Seg>();
        seg_model_->Initialize();
        seg_model_->SetDataStore(data_store_);
        data_store_->Ready(1, org_image_height, org_image_width, 3, CV_8UC3, seg_model_->input_dim_->d[2], seg_model_->input_dim_->d[3], 3);
        det_model_ = nullptr;
    }
    else
    {
        det_model_ = std::make_shared<Yolov8Det>();
        det_model_->Initialize();
        det_model_->SetDataStore(data_store_);
        data_store_->Ready(1, org_image_height, org_image_width, 3, CV_8UC3, det_model_->input_dim_->d[2], det_model_->input_dim_->d[3], 3);
        seg_model_ = nullptr;
    }

}

void AISTT::Release() {
    std::cout << "AISTT::Release()" << std::endl;

    if(camera_!=nullptr) camera_->Release();
    if(det_model_!= nullptr) det_model_->Release();
    if(seg_model_!= nullptr) seg_model_->Release();
    if(data_store_!= nullptr) data_store_->Release();
}

void AISTT::Run(){
    std::cout << "AISTT::Run()" << std::endl;
    if(det_model_== nullptr) {
        RunSeg();
        return;
    }
    while(true)
    {
        image_ = camera_->GetFrame();
        timer_.start();
        data_store_->SetOrgImage(image_);
        // det_model_->PreProcessFunc(&image_);
        det_model_->PreProcessFunc();
        sample::gLogInfo << "PreProcess time: " << timer_.end() << "ms" << std::endl;
        timer_.start();
        det_model_->Inference();
        sample::gLogInfo << "Inference time: " << timer_.end() << "ms" << std::endl;
        timer_.start();
        det_model_->PostProcessFunc();
        sample::gLogInfo << "PostProcess time: " << timer_.end() << "ms" << std::endl;
        // det_model_->Debug();
        if(!det_model_->Draw())
            break;
    }
}

void AISTT::RunSeg(){
    std::cout << "AISTT::RunSeg()" << std::endl;
    while(true)
    {
        image_ = camera_->GetFrame();
        timer_.start();
        data_store_->SetOrgImage(image_);
        seg_model_->PreProcessFunc();
        sample::gLogInfo << "PreProcess time: " << timer_.end() << "ms" << std::endl;
        timer_.start();
        seg_model_->Inference();
        sample::gLogInfo << "Inference time: " << timer_.end() << "ms" << std::endl;
        timer_.start();
        seg_model_->PostProcessFunc();
        sample::gLogInfo << "PostProcess time: " << timer_.end() << "ms" << std::endl;
        //seg_model_->Debug();
        if(!seg_model_->Draw())
            break;
    }
}