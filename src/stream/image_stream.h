//
// Created by gopizza on 2024-06-04.
//

#ifndef WINTRT_IMAGE_STREAM_H
#define WINTRT_IMAGE_STREAM_H

#include "go_camera.h"

class ImageStream : public InputStream{
public:
    void Initialize(std::string image_path, int height=1080, int width=1920);
    void Release() override;

    cv::Mat GetFrame() override;
    void Visualize() override;
private:
    cv::Mat image;
};



#endif //WINTRT_IMAGE_STREAM_H
