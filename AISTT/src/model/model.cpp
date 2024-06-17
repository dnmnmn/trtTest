//
// Created by gopizza on 2024-06-14.
//

#include "model.h"
#include "../../common/logger.h"
#include <iostream>
#include <fstream>

void Model::Initialize() {
    std::cout << "Yolov8Seg Initialize" << std::endl;
}

void Model::Release() {
    std::cout << "Yolov8Seg Release" << std::endl;
    preprocess_->release();
    postprocess_->release();
    input_dim_.reset();
    output_dims_.reset();
    engine_->Release();
}



void Model::Inference() {
    //timer_.start();
    //buffer_->cpyInputToDevice();
    engine_->Inference();
    buffer_->cpyOutputToHost(output_index_vector[0]);
    //sample::gLogInfo << "Inference time: " << timer_.end() << "ms" << std::endl;
}

void Model::PreProcessFunc() {
    //timer_.start();
    // std::cout << "Yolov8Seg PreProcess" << std::endl;
    auto data_store = data_store_.lock();
    preprocess_->gpu_resize((uchar*)data_store->gpu_org_image_data_, (uchar*)data_store->gpu_resize_image_data_, data_store->org_image_height_, data_store->org_image_width_, 3);
    preprocess_->gpu_nhwc_bgr_to_nchw_rgb((unsigned char*)data_store->gpu_resize_image_data_, (float*)engine_->GetInputGpuData(), data_store->image_height_, data_store->image_width_, 3, engine_->GetBufferStream());
    //sample::gLogInfo << "Preprocess time: " << timer_.end() << "ms" << std::endl;
}

void Model::PreProcessFunc(cv::Mat* image){
    std::cout << "Model::PreProcess" << std::endl;
}

void Model::PostProcessFunc() {
    std::cout << "Model::PostProcess" << std::endl;
}

bool Model::Draw() {
    std::cout << "Model::Draw" << std::endl;
    return true;
}

void Model::Debug() {
    std::cout << "Model::Debug" << std::endl;
    postprocess_->debug(buffer_->getOutputTensor(output_index_vector[0]));
}

void Model::SaveText(float* _data, int _size, std::string name) {
    std::string wPath = name;

    std::ofstream writeFile(wPath);
    _size /= sizeof(float);
    int h = _size/100;
    int index = 0;
    if (writeFile.is_open())
    {
        for(int i = 0; i < h; i++)
        {
            for(int j = 0; j < 100; j++)
            {
                float val = round(_data[index++] * 100)/100;
                val = val < 0 ? 0 : val;
                writeFile << val << " ";
            }
            writeFile << "\n";
            writeFile << "\n";
        }
        writeFile.close();
    }
}