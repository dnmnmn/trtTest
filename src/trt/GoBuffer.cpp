//
// Created by gopizza on 2024-06-03.
//
#include "GoBuffer.h"
#include <cuda_runtime_api.h>
#include <memory>
#include "NvInfer.h"
#include <opencv2/opencv.hpp>

// using namespace gotrt;

void gotrt::GoBuffer::initialize(int _batch_size,
                          const std::shared_ptr<nvinfer1::Dims32> _input_dim,
                          const std::shared_ptr<std::vector<nvinfer1::Dims32>> _output_dims)
{
    std::cout << "GoBuffer::initialize()" << std::endl;
    cudaStreamCreate(&stream);
    batch_size = _batch_size;

    input_tensor.channel = _input_dim->d[1];
    input_tensor.height = _input_dim->d[2];
    input_tensor.width = _input_dim->d[3];
    input_tensor.tensor_size = batch_size * input_tensor.channel * input_tensor.height * input_tensor.width * sizeof(float);
    input_tensor.cpu_tensor = new float[input_tensor.tensor_size];
    cudaMallocAsync(&input_tensor.gpu_tensor, input_tensor.tensor_size, stream);
    mDeviceBindings.push_back(input_tensor.gpu_tensor);

    output_tensor.resize(_output_dims->size());
    for(int i = 0; i < output_tensor.size(); i++)
    {
        output_tensor[i].channel = (*_output_dims)[i].d[0];
        output_tensor[i].height = (*_output_dims)[i].d[1];
        output_tensor[i].width = (*_output_dims)[i].d[2];
        output_tensor[i].tensor_size = batch_size * output_tensor[i].channel * output_tensor[i].height * output_tensor[i].width * sizeof(float);
        output_tensor[i].cpu_tensor = new float[output_tensor[i].tensor_size];
        cudaMallocAsync(&output_tensor[i].gpu_tensor, output_tensor[i].tensor_size, stream);
        mDeviceBindings.push_back(output_tensor[i].gpu_tensor);
    }
}

void gotrt::GoBuffer::release() {
    std::cout << "GoBuffer::release()" << std::endl;
    delete[] input_tensor.cpu_tensor;
    cudaFree(input_tensor.gpu_tensor);
    for(int i = 0; i < output_tensor.size(); i++)
    {
        delete[] output_tensor[i].cpu_tensor;
        cudaFree(output_tensor[i].gpu_tensor);
    }
}

void gotrt::GoBuffer::ready()
{
    std::cout << "GoBuffer::ready()" << std::endl;

}

void gotrt::GoBuffer::cpyInputToDevice()
{
    //std::cout << "GoBuffer::copyInputToDevice()" << std::endl;
    cudaMemcpyAsync(input_tensor.gpu_tensor, input_tensor.cpu_tensor, input_tensor.tensor_size, cudaMemcpyHostToDevice, stream);
}

void gotrt::GoBuffer::cpyOutputToHost(int index) {
    // std::cout << "GoBuffer::copyOutputToHost()" << std::endl;
    cudaMemcpyAsync(output_tensor[index].cpu_tensor, output_tensor[index].gpu_tensor, output_tensor[index].tensor_size, cudaMemcpyDeviceToHost, stream);
}

float* gotrt::GoBuffer::getInputTensor() {
    return input_tensor.cpu_tensor;
}

float* gotrt::GoBuffer::getOutputTensor(int index) {
    return output_tensor[index].cpu_tensor;
}

std::vector<void*> gotrt::GoBuffer::getDeviceBindings()
{
    return mDeviceBindings;
}