//
// Created by gopizza on 2024-06-04.
//

#include "video_stream.h"

void VideoStream::Initialize(std::string _video_path, int _height, int _width) {
    cap = cv::VideoCapture(_video_path);
    if(!cap.isOpened()) {
        std::cerr << "Error: VideoStream::initialize() - Could not open video file: " << _video_path << std::endl;
        return;
    }
    height_ = _height;
    width_ = _width;
    std::cout << "VideoStream::initialize()" << std::endl;
}

void VideoStream::Release() {
    std::cout << "VideoStream::release()" << std::endl;
    cap.release();
    org_image.release();
    image.release();
}

cv::Mat VideoStream::GetFrame() {
    if(cap.isOpened())
    {
        cap >> org_image;
        cv::resize(org_image, image, cv::Size(width_, height_));
        return image;
    }
    return cv::Mat();
}

void VideoStream::Visualize() {
    std::cout << "VideoStream::visualize()" << std::endl;
    if (!cap.isOpened()) {
        std::cerr << "Error: VideoStream::visualize() - Video file is not opened" << std::endl;
        return ;
    }
    while(true) {
        cap >> org_image;
        if(image.empty()) {
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