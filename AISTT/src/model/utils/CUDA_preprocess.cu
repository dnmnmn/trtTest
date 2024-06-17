#include "cuda.h"
#include "preprocess.h"
#include <cstdio>
#include "nppi.h"


__global__ void cuda_kernel_convert_uchar_to_norm_float_nchw_with_rgb(unsigned char* in, float* out, int in_size, int height, int width, int channel)
{
    unsigned int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < in_size)
    {
        unsigned int stride = height * width * channel;
        unsigned int b = index / stride;
        unsigned int stride_b = b * stride;
        unsigned int stride_y = width * channel;
        unsigned int y = (index - stride_b) / stride_y;
        unsigned int stride_x = y * stride_y;
        unsigned int x = (index - stride_b - stride_x) / channel;
        unsigned int d = index - stride_b - stride_x - (x * channel);
        //in = NHWC(BGR)
        //out = NCHW(RGB)
        unsigned int tmp_d = (channel - 1) - d;
        unsigned int out_index = (b * channel * height * width) + (tmp_d * height * width) + (y * width) + x;
        out[out_index] = ((float)in[index]) / 255.0f;
    }
}

void PreProcess::gpu_nhwc_bgr_to_nchw_rgb(uchar *src, float *dst, int _height, int _width, int _channel, cudaStream_t stream_) {
    dim3 blockDim(16, 16);
    dim3 gridDim((_width * _height * _channel + blockDim.x - 1) / blockDim.x, batch_size);
    int tensor_size = _height * _width * _channel;
    cuda_kernel_convert_uchar_to_norm_float_nchw_with_rgb << < gridDim, blockDim, 0, stream_ >> > (
            (unsigned char*)src,
            (float*)dst,
            tensor_size,
            _height, _width, _channel
    );
    cudaStreamSynchronize(stream_);
}

__global__ void cuda_kernel_convert_norm_to_nchw_with_rgb(float* in, float* out, int in_size, int height, int width, int channel)
{
    unsigned int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < in_size)
    {
        unsigned int stride = height * width * channel;
        unsigned int b = index / stride;
        unsigned int stride_b = b * stride;
        unsigned int stride_y = width * channel;
        unsigned int y = (index - stride_b) / stride_y;
        unsigned int stride_x = y * stride_y;
        unsigned int x = (index - stride_b - stride_x) / channel;
        unsigned int d = index - stride_b - stride_x - (x * channel);
        //in = NHWC(BGR)
        //out = NCHW(RGB)
        unsigned int tmp_d = (channel - 1) - d;
        unsigned int out_index = (b * channel * height * width) + (tmp_d * height * width) + (y * width) + x;
        out[out_index] = ((float)in[index]) / 255.0f;
    }
}

void PreProcess::gpu_nhwc_bgr_to_nchw_rgb(float *src, float *dst, int _height, int _width, int _channel, cudaStream_t stream_) {
    dim3 blockDim(16, 16);
    dim3 gridDim((_width * _height + blockDim.x - 1) / blockDim.x, batch_size);
    int tensor_size = _height * _width * _channel;
    cuda_kernel_convert_norm_to_nchw_with_rgb << < gridDim, blockDim, 0, stream_ >> > (
            (float*)src,
            (float*)dst,
            tensor_size,
            height, width, _channel
    );
}

void PreProcess::gpu_resize(float* src, float* dst, int _height, int _width, int _channel)
{
    int src_step = org_width * _channel * sizeof(float);
    NppiSize src_size = { org_width, org_height };
    NppiRect src_roi = { 0, 0, org_width, org_height };
    int dst_step = width * _channel * sizeof(float);
    NppiSize dst_size = { width, height };
    NppiRect dst_roi = { 0, 0, width, height };

    nppiResize_32f_C3R(
            //nppiResize_32f_C3R(
            src, src_step, src_size, src_roi,
            dst, dst_step, dst_size, dst_roi,
            NPPI_INTER_CUBIC   // interpolation method[0:NN, 1:Linear, 2:Cubic]
    );
}

void PreProcess::gpu_resize(uchar* src, uchar* dst, int _height, int _width, int _channel)
{
    int src_step = org_width * _channel * sizeof(uchar);
    NppiSize src_size = { org_width, org_height };
    NppiRect src_roi = { 0, 0, org_width, org_height };
    int dst_step = width * _channel * sizeof(uchar);
    NppiSize dst_size = { width, height };
    NppiRect dst_roi = { 0, 0, width, height };

    nppiResize_8u_C3R(
            src, src_step, src_size, src_roi,
            dst, dst_step, dst_size, dst_roi,
            NPPI_INTER_CUBIC   // interpolation method[0:NN, 1:Linear, 2:Cubic]
    );
}

