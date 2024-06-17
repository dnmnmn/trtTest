#include "../../model/utils/postprocess.h"

__global__ void cuda_kernel_inverse_matrix(float* src, float* dst, int size, int width, int height)
{
    unsigned int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size)
    {
        unsigned int b = index / (width * height);
        unsigned int y = (index % (width * height)) / width;
        unsigned int x = (index % (width * height)) % width;
        unsigned int out_index = (b * width * height) + (x * height) + y;
        dst[out_index] = src[index];
    }
}

void PostProcess::ConvertGpuInverseTensor(gotrt::Tensor* src, gotrt::Tensor* dst, cudaStream_t stream)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((dst->width * dst->height * dst->channel + blockDim.x - 1) / blockDim.x, dst->batch);
    cuda_kernel_inverse_matrix << < gridDim, blockDim, 0, stream >> > (
            (float*)src->gpu_tensor,
            (float*)dst->gpu_tensor,
            dst->tensor_size / sizeof(float),
            src->width,
            src->height
    );
    cudaMemcpyAsync(dst->cpu_tensor, dst->gpu_tensor, dst->tensor_size, cudaMemcpyDeviceToHost, stream);
}

__global__ void cuda_kernel_class_score_matrix(float* src, float* dst, int start, int class_num, int width, float conf)
{
    unsigned int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (start < index && index < start + width)
    {
        float max = 0.f;
        unsigned int max_index = 0;
        for (int i = 0; i < class_num; i++)
        {
            if (max < src[index + (i * width)])
            {
                max = src[index + (i * width)];
                max_index = i;
            }
        }
        dst[index] = max > conf ? max : 0.0f;
        dst[index + width] = max_index;
    }
}

void PostProcess::ConvertGpuClassScoreTensor(gotrt::Tensor* src, gotrt::Tensor* dst, cudaStream_t stream, int start, int class_num, float conf_threshold)
{
    cudaMemcpyAsync(dst->gpu_tensor, src->gpu_tensor, dst->tensor_size, cudaMemcpyDeviceToDevice, stream);
    unsigned int size = src->width * start;
    dim3 blockDim(16, 16);
    dim3 gridDim((src->width * src->height * src->channel + blockDim.x - 1) / blockDim.x, src->batch);
    cuda_kernel_class_score_matrix << < gridDim, blockDim, 0, stream >> > (
            (float*)src->gpu_tensor,
            (float*)dst->gpu_tensor,
            size,
            class_num,
            src->width,
            conf_threshold
    );
    cudaMemcpyAsync(dst->cpu_tensor, dst->gpu_tensor, dst->tensor_size, cudaMemcpyDeviceToHost, stream);
}