//
// Created by gopizza on 2024-05-24.
//

#ifndef WINTRT_GO_CAMERA_H
#define WINTRT_GO_CAMERA_H

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
//#include "depthai/depthai.hpp"

class InputStream {
public:
    InputStream() {};
    ~InputStream() {};
    void Initialize(int _height=1080, int _width=1920);
    virtual void Release();

    virtual cv::Mat GetFrame();
    virtual void Visualize();
    void ShowFrame();
protected:
    int32_t height_;
    int32_t width_;
};





#endif //WINTRT_GO_CAMERA_H
