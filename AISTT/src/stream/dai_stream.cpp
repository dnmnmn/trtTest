//
// Created by gopizza on 2024-06-04.
//RGB888p

#include "dai_stream.h"

void DAIStream::Initialize(std::string _file_name, int height, int width)
{
    // Define source and output
    camRgb = pipeline.create<dai::node::Camera>();
    xoutVideo = pipeline.create<dai::node::XLinkOut>();
    xoutVideo->setStreamName(_file_name);

    // Properties
    camRgb->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    camRgb->setFps(30.0f);
    // camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRgb->setVideoSize(width, height);
    // camRgb->setColorOrder(dai::ColorCameraProperties::ColorOrder::RGB);
    xoutVideo->input.setBlocking(false);
    xoutVideo->input.setQueueSize(1);
    // Linking
    camRgb->video.link(xoutVideo->input);
    // Connect to device and start data
    device = std::make_shared<dai::Device>(pipeline);
    video = device->getOutputQueue("video");
}

void DAIStream::Release() {
    std::cout << "Camera::release()" << std::endl;
    video.reset();
    device.reset();
    xoutVideo.reset();
    camRgb.reset();
}

cv::Mat DAIStream::GetFrame() {
    // std::cout << "Camera::get_frame()" << std::endl;
    auto videoIn = video->get<dai::ImgFrame>();
    return videoIn->getCvFrame();
}

void DAIStream::Visualize() {
    std::cout << "Camera::camera_test()" << std::endl;
    while(true) {
        auto videoIn = video->get<dai::ImgFrame>();

        // Get BGR frame from NV12 encoded video frame to show with opencv
        // Visualizing the frame on slower hosts might have overhead
        cv::imshow("video", videoIn->getCvFrame());

        int key = cv::waitKey(1);
        if(key == 'q' || key == 'Q') {
            return ;
        }
    }
    return ;
}
