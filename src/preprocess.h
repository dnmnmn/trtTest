//
// Created by gopizza on 2024-05-30.
//

#ifndef WINTRT_PREPROCESS_H
#define WINTRT_PREPROCESS_H

#include <cstdint>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class PreProcess {
public:
    PreProcess() {};
    ~PreProcess() {};
    void initialize(std::shared_ptr<nvinfer1::Dims32> _input_dim,
                    std::shared_ptr<std::vector<nvinfer1::Dims32>> _output_dims,
                    int32_t _org_height, int32_t _org_width);
    void release();

    void* opencv_preprocess(cv::Mat* image);
    void* crop(int x1, int y1, int x2, int y2);

    void resize(cv::Mat* src, cv::Mat* dst, int resize_height, int resize_width);
    void hwc2chw(float* src, float* dst, int c, int h, int w);
    void hwc2chw(uchar* src, uchar* dst, int c, int h, int w);
    void hwc2chw(cv::Mat* src, cv::Mat* dst);
    void chw2hwc(cv::Mat* src, cv::Mat* dst, int c, int h, int w);
    void chw2hwc(float* src, float* dst, int c, int h, int w);
    void chw2hwc(uchar* src, uchar* dst, int c, int h, int w);
    // void chw2hwc(cv::InputArray src, cv::OutputArray dst);
    void normalize(uint8_t* src, float* dst, int c, int h, int w);
    void normalize(cv::Mat* src, cv::Mat* dst);
    void denormalize(float* src, uint8_t* dst, int c, int h, int w);
    void denormalize(cv::Mat* src, cv::Mat* dst);
    void rgb2bgr(uint8_t* src, uint8_t* dst, int h, int w, int layout = int(layout_e::CHW));
    void rgb2bgr(float* src, float* dst, int h, int w, int layout = int(layout_e::CHW));
    void rgb2bgr(cv::Mat* src, cv::Mat* dst);
    void bgr2rgb(uint8_t* src, uint8_t* dst, int h, int w, int layout = int(layout_e::CHW));
    void bgr2rgb(float* src, float* dst, int h, int w, int layout = int(layout_e::CHW));
    void bgr2rgb(cv::Mat* src, cv::Mat* dst);
    void debug(float* data, std::string name, int c, int h, int w, int layout = int(layout_e::CHW));
private:
    // cv::Mat* resize_image_;
    cv::Mat* normalize_image_;
    cv::Mat* denormalize_image_;
    cv::Mat* hwc_image_;
    cv::Mat* chw_image_;
    cv::Mat* bgr_image_;
    cv::Mat* rgb_image_;
    cv::Mat* crop_image_;
    cv::Mat* crop_chw_image_;
    std::weak_ptr<nvinfer1::Dims32> input_dims_;
    std::weak_ptr<std::vector<nvinfer1::Dims32>> output_dims_;
public:
    cv::Mat* resize_image_;
    int width = 640;
    int height = 640;
    int org_width = 1920;
    int org_height = 1080;

    enum layout_e {
        HWC = 0,
        CHW = 1
    }layout_e;
};

#endif //WINTRT_PREPROCESS_H
