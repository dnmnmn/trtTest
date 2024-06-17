//
// Created by gopizza on 2024-06-14.
//

#include "data_store.h"

void DataStore::Initialize() {
    std::cout << "DataStore::Initialize()" << std::endl;
    org_image_ = std::make_shared<cv::Mat>();

}

void DataStore::Release() {
    std::cout << "DataStore::Release()" << std::endl;
    cudaFree(gpu_org_image_data_);
    cudaFree(gpu_resize_image_data_);
}

void DataStore::Ready(int _batch_size, int _org_height, int _org_width, int _org_channel, int _type, int _height, int _width, int _channel)
{
    std::cout << "DataStore::Ready()" << std::endl;
    org_image_height_ = _org_height;
    org_image_width_ = _org_width;
    org_image_channels_ = _org_channel;
    org_image_type = _type;

    image_height_ = _height;
    image_width_ = _width;
    image_channels_ = _channel;
    image_size_ = _batch_size * _height * _width * _channel * sizeof(float);

    if(_type == CV_8UC3)
    {
        org_image_size_ = org_image_width_ * org_image_height_ * org_image_channels_ * sizeof(uchar);
    }
    else if(_type == CV_32FC3)
    {
        org_image_size_ = org_image_width_ * org_image_height_ * org_image_channels_ * sizeof(float);
    }
    else std::cout << "DataStore::Ready() : Invalid image type" << std::endl;

    org_image_->create(org_image_height_, org_image_width_, org_image_type);
    cudaMalloc(&gpu_org_image_data_, org_image_size_);
    cudaMalloc(&gpu_resize_image_data_, image_size_);
    return;
}

void DataStore::SetOrgImage(cv::Mat _input) {
    memcpy(org_image_->data, _input.data, org_image_size_);
    cudaMemcpy(gpu_org_image_data_, org_image_->data, org_image_size_, cudaMemcpyHostToDevice);
}