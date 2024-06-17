//
// Created by gopizza on 2024-06-14.
//

#ifndef WINTRT_MODEL_H
#define WINTRT_MODEL_H

#include "../trt/go_engine.h"
#include "utils/postprocess.h"
#include "utils/preprocess.h"
#include "Timer/Timer.h"
#include <opencv2/opencv.hpp>
#include "../data/data_store.h"
#include <memory>

class Model {
public:
    Model() {};
    ~Model() {};
    virtual void Initialize();
    void Release();

    void Inference();
    void PreProcessFunc();
    virtual void PreProcessFunc(cv::Mat* image);
    virtual void PostProcessFunc();

    virtual bool Draw();
    void Debug();
    void SaveText(float* _data, int _size, std::string name);
    void SetDataStore(std::shared_ptr<DataStore> _data_store){
        data_store_ = _data_store;
    };

protected:
    Timer timer_;
    std::weak_ptr<DataStore> data_store_;
    std::shared_ptr<PreProcess> preprocess_;
    std::shared_ptr<PostProcess> postprocess_;
    std::shared_ptr<gotrt::GoEngine> engine_;
    std::shared_ptr<gotrt::GoBuffer> buffer_;
public:
    std::vector<int32_t> output_index_vector;
    std::shared_ptr<nvinfer1::Dims32> input_dim_;
    std::shared_ptr<std::vector<nvinfer1::Dims32>> output_dims_;
};


#endif //WINTRT_MODEL_H
