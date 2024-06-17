//
// Created by gopizza on 2024-06-04.
//

#include "camera_stream.h"

void CameraStream::Initialize(std::string _file_name, int _height, int _width) {
    height_ = _height;
    width_ = _width;
    cap = cv::VideoCapture(1);
    if(!cap.isOpened()) {
        std::cerr << "Error: CameraStream::initialize() - webcam not exist" << std::endl;
        return;
    }
    std::cout << "CameraStream::initialize()" << std::endl;
}

void CameraStream::Release() {
    std::cout << "CameraStream::release()" << std::endl;
    cap.release();
    org_image.release();
    image.release();
}

cv::Mat CameraStream::GetFrame() {
    if(cap.isOpened())
    {
        cap >> org_image;
        cv::resize(org_image, image, cv::Size(width_, height_));
        return image;
    }
    return cv::Mat();
}

void CameraStream::Visualize() {
    std::cout << "CameraStream::Visualize()" << std::endl;
    if (!cap.isOpened()) {
        std::cerr << "Error: CameraStream::Visualize() - webcam not exist" << std::endl;
        return ;
    }
    while(true) {
        cap >> org_image;
        if(org_image.empty()) {
            std::cerr << "Error: VideoStream::visualize() - Could not read frame from video file" << std::endl;
            break;
        }
        cv::resize(org_image, image, cv::Size(width_, height_));
        cv::imshow("video", image);
        int key = cv::waitKey(1);
        if(key == 'q' || key == 'Q') {
            return ;
        }
    }
    return ;
}