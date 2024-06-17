//
// Created by gopizza on 2024-05-24.
//

#ifndef WINTRT_GO_CAMERA_H
#define WINTRT_GO_CAMERA_H

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "../data/data_store.h"
//#include "depthai/depthai.hpp"

class InputStream {
public:
    InputStream() {};
    ~InputStream() {};
    virtual void Initialize(std::string _file_name="InputStream", int _height=1080, int _width=1920);
    virtual void Release();

    virtual cv::Mat GetFrame();
    virtual void Visualize();
    void ShowFrame();
    void SetDataStore(std::shared_ptr<DataStore> _data_store){
        data_store_ = _data_store;
    };
protected:
    int32_t height_;
    int32_t width_;
    std::weak_ptr<DataStore> data_store_;
};





#endif //WINTRT_GO_CAMERA_H
