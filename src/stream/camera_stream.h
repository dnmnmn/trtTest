//
// Created by gopizza on 2024-06-04.
//

#ifndef WINTRT_CAMERA_STREAM_H
#define WINTRT_CAMERA_STREAM_H

#include "go_camera.h"

class CameraStream : public InputStream{
public:
    void Initialize(int _height=1080, int _width=1920);
    void Release() override;

    cv::Mat GetFrame() override;
    void Visualize() override;
private:
    cv::VideoCapture cap;
    cv::Mat org_image;
    cv::Mat image;
};


#endif //WINTRT_CAMERA_STREAM_H
