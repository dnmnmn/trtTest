//
// Created by gopizza on 2024-05-24.
//
#include "input_stream.h"
#include <iostream>

void InputStream::Initialize(std::string _file_name, int _height, int _width) {
    std::cout << "InputStream::Initialize()" << std::endl;
    height_ = _height;
    width_ = _width;
}

void InputStream::Release() {
    std::cout << "InputStream::Release()" << std::endl;
}

void InputStream::ShowFrame() {
    std::cout << "InputStream::ShowFrame()" << std::endl;
}

cv::Mat InputStream::GetFrame() {
    std::cout << "InputStream::GetFrame()" << std::endl;
    return cv::Mat();
}

void InputStream::Visualize() {
    std::cout << "InputStream::Visualize()" << std::endl;
}




