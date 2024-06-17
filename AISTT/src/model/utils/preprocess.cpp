//
// Created by gopizza on 2024-05-30.
//
#include "preprocess.h"
#include <fstream>

void PreProcess::initialize(std::shared_ptr<nvinfer1::Dims32> _input_dim,
                            std::shared_ptr<std::vector<nvinfer1::Dims32>> _output_dims,
                            int32_t _org_height, int32_t _org_width) {
    input_dims_ = _input_dim;
    output_dims_ = _output_dims;
    height = _input_dim->d[2];
    width = _input_dim->d[3];
    org_height = _org_height;
    org_width = _org_width;
    resize_image_ = new cv::Mat(height, width, CV_8UC3);
    hwc_image_ = new cv::Mat(height, width, CV_8UC3);
    chw_image_ = new cv::Mat(height, width, CV_32FC3);
    rgb_image_ = new cv::Mat(height, width, CV_8UC3);
    normalize_image_ = new cv::Mat(height, width, CV_32FC3);
    crop_image_ = new cv::Mat(640, 640, CV_32FC3);
    crop_chw_image_ = new cv::Mat(640, 640, CV_32FC3);
    std::cout << "=============PreProcess::initialize=============" << std::endl;
    printf("width: %d, height: %d, org_width: %d, org_height: %d\n", width, height, org_width, org_height);
}

void PreProcess::release() {
    delete resize_image_;
    delete hwc_image_;
    delete chw_image_;
    delete rgb_image_;
    delete normalize_image_;
    delete crop_image_;
    delete crop_chw_image_;
    std::cout << "PreProcess::release()" << std::endl;
}

void* PreProcess::opencv_preprocess(cv::Mat* image)
{
    resize(image, resize_image_, height, width);
    bgr2rgb(resize_image_, rgb_image_);

    normalize(rgb_image_, normalize_image_);
    hwc2chw(normalize_image_, chw_image_);

//    bgr2rgb(image, rgb_image);
//    normalize(rgb_image, normalize_image);
//    resize(normalize_image, resize_image, height, width);
//    hwc2chw(resize_image, chw_image);

    return chw_image_->data;
}

void* PreProcess::crop(int x1, int y1, int x2, int y2)
{
    x1 = x1 * org_width / width;
    y1 = y1 * org_height / height;
    x2 = x2 * org_width / width;
    y2 = y2 * org_height / height;
    cv::Mat Roi(*normalize_image_, cv::Rect(x1, y1, x2 - x1, y2 - y1));
    Roi.copyTo(*crop_image_);
    cv::resize(*crop_image_, *crop_image_, cv::Size(640, 640));
    hwc2chw(crop_image_, crop_chw_image_);

    return crop_chw_image_->data;
}

void PreProcess::resize(cv::Mat* srcImage, cv::Mat* dstImage, int resize_width, int resize_height)
{
    cv::resize(*srcImage, *dstImage, cv::Size(resize_width, resize_height));
}

void PreProcess::hwc2chw(float* src, float* dst, int channel, int height, int width) {
    int hw_size = height * width;
    int wc_size = width * channel;
    for(int h = 0; h < height; h++)
    {
        for(int w = 0; w < width; w++)
        {
            for(int c = 0; c < channel; c++)
            {
                dst[c * hw_size + h * width + w] = src[h * wc_size + w * channel + c];
            }
        }
    }
}

void PreProcess::hwc2chw(uchar* src, uchar* dst, int channel, int height, int width) {
    int hw_size = height * width;
    int wc_size = width * channel;
    for(int h = 0; h < height; h++)
    {
        for(int w = 0; w < width; w++)
        {
            for(int c = 0; c < channel; c++)
            {
                dst[c * hw_size + h * width + w] = src[h * wc_size + w * channel + c];
            }
        }
    }
}

void PreProcess::hwc2chw(cv::Mat* src, cv::Mat* dst) {
    std::vector<cv::Mat> channels;
    cv::split(*src, channels);

    // Stretch one-channel images to vector
    for (auto &img : channels) {
        img = img.reshape(1, 1);
    }

    // Concatenate three vectors to one
    cv::hconcat( channels, *dst );
}

void PreProcess::chw2hwc(cv::Mat* src, cv::Mat* dst, int channel, int height, int width) {
    int hw_size = height * width;
    int wc_size = width * channel;
    for(int c = 0; c < channel; c++)
    {
        for(int h = 0; h < height; h++)
        {
            for(int w = 0; w < width; w++)
            {
                dst->data[h * wc_size + w * channel + c] = src->data[c * hw_size + h * width + w];
            }
        }
    }
}

void PreProcess::chw2hwc(float* src, float* dst, int channel, int height, int width) {
    for(int h = 0; h < height; h++)
    {
        for(int w = 0; w < width; w++)
        {
            for(int c = 0; c < channel; c++)
            {
                dst[h * width * channel + w * channel + c] = src[c * height * width + h * width + w];
            }
        }
    }
}

void PreProcess::chw2hwc(uchar* src, uchar* dst, int channel, int height, int width) {
    for(int h = 0; h < height; h++)
    {
        for(int w = 0; w < width; w++)
        {
            for(int c = 0; c < channel; c++)
            {
                dst[h * width * channel + w * channel + c] = src[c * height * width + h * width + w];
            }
        }
    }
}

void PreProcess::normalize(uint8_t *src, float *dst, int c, int h, int w)
{
    for(int i = 0; i < h * w * c; i++)
    {
        dst[i] = (float)src[i] / 255.0f;
    }
}
void PreProcess::normalize(cv::Mat* src, cv::Mat* dst)
{
    src->convertTo(*dst, CV_32FC3, 1.0/255.0);
}

void PreProcess::denormalize(float *src, uint8_t *dst, int c, int h, int w)
{
    for(int i =0; i < h * w * c; i++)
    {
        dst[i] = uint8_t(src[i] * 255);
    }
}
void PreProcess::denormalize(cv::Mat *src, cv::Mat *dst)
{
    src->convertTo(*dst, CV_8UC3, 255);
}
void PreProcess::rgb2bgr(uint8_t *src, uint8_t *dst, int h, int w, int layout) {
    if(layout == (int)layout_e::CHW)
    {
        int size = h * w * sizeof(uint8_t);
        memcpy(dst + size * 2, src, size);
        memcpy(dst + size, src + size, size);
        memcpy(dst, src + size * 2 , size);
//        for(int i = 0; i < h * w; i++)
//        {
//            // Red
//            dst[size * 2 + i] = src[i];
//            // Green
//            dst[size + i] = src[size + i];
//            // Blue
//            dst[i] = src[size * 2 + i];
//        }

    }
    else
    {
        for(int i = 0; i < h * w; i++)
        {
            // Red
            dst[i * 3 + 2] = src[i * 3];
            // Green
            dst[i * 3 + 1] = src[i * 3 + 1];
            // Blue
            dst[i * 3] = src[i * 3 + 2];
        }
    }
}

void PreProcess::rgb2bgr(float *src, float *dst, int h, int w, int layout)
{
    if(layout == (int)layout_e::CHW)
    {
        int size = h * w * sizeof(uint8_t);
        memcpy(dst + size * 2, src, size);
        memcpy(dst + size, src + size, size);
        memcpy(dst, src + size * 2 , size);
//        for(int i = 0; i < h * w; i++)
//        {
//            // Red
//            dst[h * w * 2 + i] = src[i];
//            // Green
//            dst[h * w + i] = src[h * w + i];
//            // Blue
//            dst[i] = src[h * w * 2 + i];
//        }
    }
    else
    {
        for(int i = 0; i < h * w; i++)
        {
            // Red
            dst[i * 3 + 2] = src[i * 3];
            // Green
            dst[i * 3 + 1] = src[i * 3 + 1];
            // Blue
            dst[i * 3] = src[i * 3 + 2];
        }
    }
}

void PreProcess::rgb2bgr(cv::Mat* src, cv::Mat* dst)
{
    cv::cvtColor(*src, *dst, cv::COLOR_RGB2BGR);
}

void PreProcess::bgr2rgb(uint8_t *src, uint8_t *dst, int h, int w, int layout) {
    if(layout == (int)layout_e::CHW)
    {
        int size = h * w * sizeof(uint8_t);
        memcpy(dst, src + size * 2 , size);
        memcpy(dst + size, src + size, size);
        memcpy(dst + size * 2, src, size);
//        for(int i = 0; i < h * w; i++)
//        {
//            // Red
//            dst[i] = src[size * 2 + i];
//            // Green
//            dst[size + i] = src[size + i];
//            // Blue
//            dst[size * 2 + i] = src[i];
//        }
    }
    else
    {
        for(int i = 0; i < h * w; i++)
        {
            // Red
            dst[i * 3] = src[i * 3 + 2];
            // Green
            dst[i * 3 + 1] = src[i * 3 + 1];
            // Blue
            dst[i * 3 + 2] = src[i * 3];
        }
    }
}

void PreProcess::bgr2rgb(float *src, float *dst, int h, int w, int layout) {
    if(layout == (int)layout_e::CHW)
    {
        int size = h * w * sizeof(uint8_t);
        memcpy(dst, src + size * 2 , size);
        memcpy(dst + size, src + size, size);
        memcpy(dst + size * 2, src, size);
//        for(int i = 0; i < h * w; i++)
//        {
//            // Red
//            dst[i] = src[h * w * 2 + i];
//            // Green
//            dst[size + i] = src[h * w + i];
//            // Blue
//            dst[size * 2 + i] = src[i];
//        }
    }
    else
    {
        for(int i = 0; i < h * w; i++)
        {
            // Red
            dst[i * 3] = src[i * 3 + 2];
            // Green
            dst[i * 3 + 1] = src[i * 3 + 1];
            // Blue
            dst[i * 3 + 2] = src[i * 3];
        }
    }
}

void PreProcess::bgr2rgb(cv::Mat* src, cv::Mat* dst)
{
    cv::cvtColor(*src, *dst, cv::COLOR_BGR2RGB);
}

void PreProcess::debug(float* data, std::string name, int channel, int height, int width, int type)
{
    std::cout << "PostProcess::debug()" << std::endl;
    std::string wPath = name + ".txt";

    std::ofstream writeFile(wPath);
    if (writeFile.is_open())
    {
        if(type==layout_e::CHW)
        {
            int tensor_size = channel * height * width;
            for (int i = 0; i < tensor_size; i++)
            {
                if(i % width == 0)
                {
                    writeFile << "\n";
                    if(i % height == 0) writeFile << "\n";
                }
                float val = data[i];
                writeFile << val << " ";
            }
        }
        if (type == layout_e::HWC)
        {
            int hw_size = height * width;
            for (int c = 0; c < channel; c++)
            {
                for(int h = 0; h < height; h++)
                {
                    for(int w = 0; w < width; w++)
                    {
                        int index = c * hw_size + h * width + w;
                        float val = data[index];
                        writeFile << val << " ";
                    }
                    writeFile << "\n";
                }
                writeFile << "\n";
            }
        }
        else
        {
            std::cout << "layout type error" << std::endl;
        }
        writeFile.close();
    }
}