//
// Created by gopizza on 2024-06-03.
//

#ifndef WINTRT_GOBUFFER_H
#define WINTRT_GOBUFFER_H

#include <vector>
#include <iostream>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "cuda.h"
#include <opencv2/opencv.hpp>

namespace gotrt{
    struct Tensor {
        float* cpu_tensor;
        void* gpu_tensor;
        int tensor_size;
        int batch;
        int channel;
        int width;
        int height;
    };

    class GoBuffer {
    public:
        GoBuffer() {};
        ~GoBuffer() {};
        void initialize(int _batch_size,
                        const std::shared_ptr<nvinfer1::Dims32> _input_dim,
                        const std::shared_ptr<std::vector<nvinfer1::Dims32>> _output_dims,
                        int _org_image_height,
                        int _org_image_width,
                        int class_num=1);
        void release();
        void ready();

        void cpyInputToDevice();
        void cpyOutputToHost(int output_tensor_index);
        void cpyOrginInputToDevice(cv::Mat* image);
        float *getInputData();
        float *getInputGpuData();
        float *getOutputData(int index);
        gotrt::Tensor *getOutputTensor(int index);
        std::vector<void *> getDeviceBindings();
    public:
        Tensor class_score_tensor;
        Tensor inverse_tensor;
        Tensor origin_input_tensor;
        cudaStream_t stream;
    private:
        int batch_size;
        int class_num_;
        Tensor input_tensor;

        std::vector<Tensor> output_tensor;
        std::vector<void*> mDeviceBindings;
    };
}

#endif //WINTRT_GOBUFFER_H