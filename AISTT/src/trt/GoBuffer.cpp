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
                          const std::shared_ptr<std::vector<nvinfer1::Dims32>> _output_dims,
                          int _org_image_height,
                          int _org_image_width,
                          int _class_num)
{
    std::cout << "GoBuffer::initialize()" << std::endl;
    cudaStreamCreate(&stream);
    batch_size = _batch_size;
    input_tensor.batch = batch_size;
    input_tensor.channel = _input_dim->d[1];
    input_tensor.height = _input_dim->d[2];
    input_tensor.width = _input_dim->d[3];
    input_tensor.tensor_size = batch_size * input_tensor.channel * input_tensor.height * input_tensor.width * sizeof(float);
    input_tensor.cpu_tensor = new float[input_tensor.tensor_size];
    cudaMallocAsync(&input_tensor.gpu_tensor, input_tensor.tensor_size, stream);
    mDeviceBindings.push_back(input_tensor.gpu_tensor);

    origin_input_tensor.batch = batch_size;
    origin_input_tensor.channel = 3;
    origin_input_tensor.height = _org_image_height;
    origin_input_tensor.width = _org_image_width;
    origin_input_tensor.tensor_size = batch_size * origin_input_tensor.channel * origin_input_tensor.height * origin_input_tensor.width * sizeof(float);
    origin_input_tensor.cpu_tensor = new float[origin_input_tensor.tensor_size];
    cudaMallocAsync(&origin_input_tensor.gpu_tensor, origin_input_tensor.tensor_size, stream);

    output_tensor.resize(_output_dims->size());
    for(int i = 0; i < output_tensor.size(); i++)
    {
        if((*_output_dims)[i].nbDims == 3)
        {
            output_tensor[i].batch = batch_size;
            output_tensor[i].channel = (*_output_dims)[i].d[0];
            output_tensor[i].height = (*_output_dims)[i].d[1];
            output_tensor[i].width = (*_output_dims)[i].d[2];
        }
        else if ((*_output_dims)[i].nbDims == 4)
        {
            output_tensor[i].batch = batch_size;
            output_tensor[i].channel = (*_output_dims)[i].d[1];
            output_tensor[i].height = (*_output_dims)[i].d[2];
            output_tensor[i].width = (*_output_dims)[i].d[3];
        }
        else if ((*_output_dims)[i].nbDims == 2)
        {
            output_tensor[i].batch = batch_size;
            output_tensor[i].channel = 1;
            output_tensor[i].height = (*_output_dims)[i].d[3];
            output_tensor[i].width = (*_output_dims)[i].d[4];
        }
        else
        {
            std::cout << "output tensor dims error" << std::endl;
        }
        output_tensor[i].tensor_size = batch_size * output_tensor[i].channel * output_tensor[i].height * output_tensor[i].width * sizeof(float);
        output_tensor[i].cpu_tensor = new float[output_tensor[i].tensor_size];
        cudaMallocAsync(&output_tensor[i].gpu_tensor, output_tensor[i].tensor_size, stream);
        cudaMemset(output_tensor[i].gpu_tensor, 0, (size_t)output_tensor[i].tensor_size);
        mDeviceBindings.push_back(output_tensor[i].gpu_tensor);
    }
    // xywhsck tensor
    class_num_ = _class_num;
    class_score_tensor.batch = batch_size;
    class_score_tensor.channel = 1;
    class_score_tensor.height = 6;
    class_score_tensor.width = output_tensor[output_tensor.size() - 1].width;
    class_score_tensor.tensor_size = batch_size * class_score_tensor.channel * class_score_tensor.height * class_score_tensor.width * sizeof(float);
    class_score_tensor.cpu_tensor = new float[class_score_tensor.tensor_size];
    cudaMallocAsync(&class_score_tensor.gpu_tensor, class_score_tensor.tensor_size, stream);
    cudaMemset(class_score_tensor.gpu_tensor, 0, (size_t)class_score_tensor.tensor_size);
    // inverse tensor
    inverse_tensor.batch = batch_size;
    inverse_tensor.channel = 1;
    inverse_tensor.height = output_tensor[output_tensor.size() - 1].width;
    inverse_tensor.width = 6;
    inverse_tensor.tensor_size = batch_size * inverse_tensor.channel * inverse_tensor.height * inverse_tensor.width * sizeof(float);
    inverse_tensor.cpu_tensor = new float[inverse_tensor.tensor_size];
    cudaMallocAsync(&inverse_tensor.gpu_tensor, inverse_tensor.tensor_size, stream);
    cudaMemset(inverse_tensor.gpu_tensor, 0, (size_t)inverse_tensor.tensor_size);
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
    delete[] inverse_tensor.cpu_tensor;
    cudaFree(inverse_tensor.gpu_tensor);

    delete[] class_score_tensor.cpu_tensor;
    cudaFree(class_score_tensor.gpu_tensor);

    delete[] origin_input_tensor.cpu_tensor;
    cudaFree(origin_input_tensor.gpu_tensor);

    output_tensor.clear();
    mDeviceBindings.clear();
    cudaStreamDestroy(stream);
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

void gotrt::GoBuffer::cpyOrginInputToDevice(cv::Mat* image) {
    //memcpy(origin_input_tensor.cpu_tensor, image->data, origin_input_tensor.tensor_size);
    cudaMemcpyAsync(origin_input_tensor.gpu_tensor, origin_input_tensor.cpu_tensor, origin_input_tensor.tensor_size, cudaMemcpyHostToDevice, stream);
}

float* gotrt::GoBuffer::getInputData() {
    return input_tensor.cpu_tensor;
}

float* gotrt::GoBuffer::getInputGpuData()
{
    return (float*)input_tensor.gpu_tensor;
}

float* gotrt::GoBuffer::getOutputData(int index) {
    return output_tensor[index].cpu_tensor;
}

gotrt::Tensor* gotrt::GoBuffer::getOutputTensor(int index)
{
    return &output_tensor[index];
}

std::vector<void*> gotrt::GoBuffer::getDeviceBindings()
{
    return mDeviceBindings;
}