//
// Created by gopizza on 2024-06-07.
//

#ifndef WINTRT_YOLOV8_DET_H
#define WINTRT_YOLOV8_DET_H

#include "../model.h"
#include "../../trt/go_engine.h"
#include "../utils/postprocess.h"
#include "../utils/preprocess.h"
#include "../utils/postprocess_det.h"
#include "Timer/Timer.h"

class Yolov8Det : public Model{
public:
    Yolov8Det() {};
    ~Yolov8Det() {};
    void Initialize();

    //void Inference();
    void PreProcessFunc(cv::Mat* image);
    void PostProcessFunc();
    bool Draw();
    //void Debug();
private:
    bool LoadEngine(std::string _model_path);
private:
//    Timer timer_;
//    std::shared_ptr<PreProcess> preprocess_;
//    std::shared_ptr<PostProcess> postprocess_;
//    std::shared_ptr<gotrt::GoEngine> engine_;
//    std::shared_ptr<gotrt::GoBuffer> buffer_;
//    std::shared_ptr<nvinfer1::Dims32> input_dim_;
//    std::shared_ptr<std::vector<nvinfer1::Dims32>> output_dims_;
private:
    bool use_obb_;
    int32_t org_image_width_;
    int32_t org_image_height_;
    int32_t input_index;
    int32_t input_tensor_size_;
    //std::vector<int32_t> output_index_vector;
    std::vector<int32_t> output_tensor_size_;
    std::vector<std::vector<float>> boxes;
};


#endif //WINTRT_YOLOV8_DET_H
