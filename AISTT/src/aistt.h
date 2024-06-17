//
// Created by gopizza on 2024-06-04.
//

#ifndef WINTRT_AISTT_H
#define WINTRT_AISTT_H
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#include "data/data_store.h"
#include "model/detection/yolov8_det.h"
#include "model/segmentation/yolov8_seg.h"
#include "model/model.h"
#include "stream/input_stream.h"
#include "stream/dai_stream.h"
#include "stream/image_stream.h"
#include "stream/video_stream.h"
#include "Timer/timer.h"

class AISTT {
public:
    AISTT() {};
    ~AISTT() {};
    void Initialize();
    void Release();
    void Run();
    void RunSeg();
private:
    std::shared_ptr<InputStream> camera_= nullptr;
    std::shared_ptr<DataStore> data_store_= nullptr;
    std::shared_ptr<Model> det_model_;
    std::shared_ptr<Model> seg_model_;
    cv::Mat image_;
    Timer timer_;
};

#endif //WINTRT_GOENGINE_H
