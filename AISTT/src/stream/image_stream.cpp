//
// Created by gopizza on 2024-06-04.
//

#include "image_stream.h"

void ImageStream::Initialize(std::string image_path, int height, int width) {
    image = cv::imread(image_path);
    if(image.empty()) {
        std::cerr << "Error: ImageStream::initialize() - Could not open image file: " << image_path << std::endl;
        return;
    }
    height_ = height;
    width_ = width;
    cv::resize(image, image, cv::Size(width, height));
    std::cout << "ImageStream::initialize()" << std::endl;
}

void ImageStream::Release() {
    std::cout << "ImageStream::release()" << std::endl;
    image.release();
}

cv::Mat ImageStream::GetFrame() {
    return image;
}

void ImageStream::Visualize() {
    std::cout << "ImageStream::visualize()" << std::endl;
    if(image.empty()) {
        std::cerr << "Error: ImageStream::visualize() - Image is empty" << std::endl;
        return ;
    }
    cv::imshow("image", image);
    cv::waitKey(0);
}