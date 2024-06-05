//
// Created by gopizza on 2024-06-04.
//

#ifndef WINTRT_VIDEO_STREAM_H
#define WINTRT_VIDEO_STREAM_H

#include "go_camera.h"

class VideoStream : public InputStream{
public:
    void Initialize(std::string _video_path, int _height=1080, int _width=1920) override;
    void Release() override;

    cv::Mat GetFrame() override;
    void Visualize() override;
private:
    cv::VideoCapture cap;
    cv::Mat org_image;
    cv::Mat image;
};


#endif //WINTRT_VIDEO_STREAM_H
