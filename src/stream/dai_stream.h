//
// Created by gopizza on 2024-06-04.
//

#ifndef WINTRT_DAI_STREAM_H
#define WINTRT_DAI_STREAM_H

#include "go_camera.h"
#include "depthai/depthai.hpp"

class DAIStream : public InputStream{
public:
    void Initialize(int height=1080, int width=1920);
    void Release() override;

    cv::Mat GetFrame() override;
    void Visualize() override;
private:
    dai::Pipeline pipeline;
    std::shared_ptr<dai::node::Camera> camRgb;
    std::shared_ptr<dai::node::XLinkOut> xoutVideo;
    std::shared_ptr<dai::Device> device;
    std::shared_ptr<dai::DataOutputQueue> video;
};



#endif //WINTRT_DAI_STREAM_H
