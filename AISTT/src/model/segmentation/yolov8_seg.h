//
// Created by gopizza on 2024-06-10.
//

#ifndef WINTRT_YOLOV8_SEG_H
#define WINTRT_YOLOV8_SEG_H

#include "../model.h"
#include "../../trt/go_engine.h"
#include "../utils/postprocess.h"
#include "../utils/postprocess_seg.h"
#include "../utils/preprocess.h"
#include "Timer/Timer.h"

class Yolov8Seg : public Model{
public:
    Yolov8Seg() {};
    ~Yolov8Seg() {};
    void Initialize();

    void Inference();
    void PreProcessFunc(cv::Mat* image);
    void PostProcessFunc();
    bool Draw();
    void Debug();
private:
    bool LoadEngine(std::string _model_path);
private:
//    std::shared_ptr<PreProcess> preprocess_;
//    std::shared_ptr<PostProcess> postprocess_;
//    std::shared_ptr<gotrt::GoEngine> engine_;
//    std::shared_ptr<gotrt::GoBuffer> buffer_;
//    std::shared_ptr<nvinfer1::Dims32> input_dim_;
//    std::shared_ptr<std::vector<nvinfer1::Dims32>> output_dims_;
private:
    //Timer timer_;
    int32_t org_image_width_;
    int32_t org_image_height_;
    int32_t input_index;
    int32_t input_tensor_size_;
    //std::vector<int32_t> output_index_vector;
    std::vector<int32_t> output_tensor_size_;
    std::vector<std::vector<float>> boxes;
};


#endif //AISMARTTOPPINGTABLE2_YOLOV8_SEG_H
