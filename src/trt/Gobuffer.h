//
// Created by gopizza on 2024-06-03.
//

#ifndef WINTRT_GOBUFFER_H
#define WINTRT_GOBUFFER_H

#endif //WINTRT_GOBUFFER_H

#include <vector>
#include <iostream>
#include "NvInfer.h"
#include <cuda_runtime_api.h>

namespace gotrt{
    struct Tensor {
        float* cpu_tensor;
        void* gpu_tensor;
        int tensor_size;
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
                        const std::shared_ptr<std::vector<nvinfer1::Dims32>> _output_dims);
        void release();
        void ready();

        void cpyInputToDevice();
        void cpyOutputToHost(int output_tensor_index);
        float *getOutputTensor(int index);
        float *getInputTensor();
        std::vector<void *> getDeviceBindings();

    private:
        int batch_size;
        Tensor input_tensor;
        std::vector<Tensor> output_tensor;
        std::vector<void*> mDeviceBindings;
        cudaStream_t stream;
    };
}