//
// Created by gopizza on 2024-06-14.
//

#ifndef AISMARTTOPPINGTABLE2_DATA_STORE_H
#define AISMARTTOPPINGTABLE2_DATA_STORE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>
#include <cuda_runtime.h>

class DataStore {
public:
    DataStore() {};
    ~DataStore() {};
    void Initialize();
    void Release();
    void Ready(int _batch_size, int _org_height, int _org_width, int _org_channel, int _type, int _height, int _width, int _channel);
    void SetOrgImage(cv::Mat _input);

    int batch_size_;

    int org_image_width_;
    int org_image_height_;
    int org_image_channels_;
    int org_image_size_;
    int org_image_type;
    std::shared_ptr<cv::Mat> org_image_;

    int image_width_;
    int image_height_;
    int image_channels_;
    int image_size_;

    // gpu memory
    void* gpu_org_image_data_;
    void* gpu_resize_image_data_;
};

#endif //AISMARTTOPPINGTABLE2_DATA_STORE_H
