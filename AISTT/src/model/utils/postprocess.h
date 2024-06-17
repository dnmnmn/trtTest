//
// Created by gopizza on 2024-05-23.
//

#ifndef WINTRT_POSTPROCESS_H
#define WINTRT_POSTPROCESS_H

#include <vector>
#include <memory>
#include "opencv2/opencv.hpp"
#include <assert.h>
#include "NvInfer.h"
#include "../../trt/Gobuffer.h"


class PostProcess {
public:
    PostProcess() {};
    ~PostProcess() {};
    void initialize(std::shared_ptr<nvinfer1::Dims32> _input_dim,
                    std::shared_ptr<std::vector<nvinfer1::Dims32>> _output_dims);
    void release();

    std::vector<std::vector<float>> weighted_boxes_fusion(const float iou_threshold=0.45f);
    std::vector<std::vector<float>> nms(const float iou_threshold=0.45f);

    void debug(gotrt::Tensor* _tensor);
    void clear();

    virtual void normalize(float* data, float conf_threshold=0.35f) = 0;
    virtual void denormalize(std::vector<std::vector<float>> _input_box, std::vector<std::vector<float>>* _output_box) = 0;

protected:
    int input_width;
    int input_height;
    int output_width;
    int output_height;
    int output_channel;
    int box_count = 0;
    const int MAX_NMS_ = 300;
    std::vector<std::vector<float>> boxes;
    std::weak_ptr<nvinfer1::Dims32> input_dim_;
    std::weak_ptr<std::vector<nvinfer1::Dims32>> output_dims_;
    float* inverse_data;
public:
    void ConvertGpuInverseTensor(gotrt::Tensor* src, gotrt::Tensor* dst, cudaStream_t stream);
    void ConvertGpuClassScoreTensor(gotrt::Tensor* src, gotrt::Tensor* dst, cudaStream_t stream, int start, int class_num, float conf_threshold=0.35f);

protected:
    std::pair<int, float> find_matching_box_fast(const std::vector<std::vector<float>>& boxes, const std::vector<float>& new_box, const float iou_threshold, int box_count);
    std::vector<float> bb_iou_array(const std::vector<std::vector<float>> boxes, const std::vector<float> new_box);
    std::vector<float> get_weighted_box(const std::vector<std::vector<float>> boxes, int conf_type, int box_count);
    float get_overlap_area(const std::vector<float>& box1, const std::vector<float>& box2);
    float get_box_area(const std::vector<float>& box);
protected:
    inline void GetInverseMatrix(const float* data, int height, int width)
    {
        for(int i =0; i < height; i++)
        {
            for(int j = 0; j < width; j++)
            {
                inverse_data[j * height + i] = data[i * width + j];
            }
        }
    };
};



#endif //WINTRT_POSTPROCESS_H