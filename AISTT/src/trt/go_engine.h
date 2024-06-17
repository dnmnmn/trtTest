//
// Created by gopizza on 2024-06-04.
//

#ifndef WINTRT_GO_ENGINE_H
#define WINTRT_GO_ENGINE_H

#include <string>
#include <memory>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "GoBuffer.h"

namespace gotrt {
    struct ModelParams
    {
        int32_t batch_size{1};              //!< Number of inputs in a batch
        int32_t dla_core{-1};               //!< Specify the DLA core to run network on.
        bool int8{false};                  //!< Allow runnning the network in Int8 mode.
        bool fp16{false};                  //!< Allow running the network in FP16 mode.
        std::string wight_path; //!< Directory paths where sample data files are stored
        int32_t input_index{0};             //!< Input index of the network
        std::vector<int32_t> output_index_vector; //!< Output index of the network
    };

    class GoEngine {
    public:
        GoEngine() {};
        ~GoEngine() {};
        void Initialize();
        void Release();

        bool LoadEngine(ModelParams _model_params,
                        std::shared_ptr<nvinfer1::Dims32> _input_dims,
                        std::shared_ptr<std::vector<nvinfer1::Dims32>> _output_dims);
        void SetBuffer(std::shared_ptr<gotrt::GoBuffer> _buffer) { buffer_ = _buffer; }
        bool Inference();
        float* GetInputData();
        float* GetInputGpuData();
        float* GetOutputData(int _index);
        cudaStream_t GetStream() { return *stream_; }
        cudaStream_t GetBufferStream() { return buffer_.lock()->stream; }
        gotrt::Tensor* GetOutputTensor(int _index);

    private:
        bool ConstructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
                              std::unique_ptr<nvinfer1::INetworkDefinition>& network,
                              std::unique_ptr<nvinfer1::IBuilderConfig>& config,
                              std::unique_ptr<nvonnxparser::IParser>& parser,
                              ModelParams model_params);
    private:
        std::unique_ptr<cudaStream_t> stream_;
        std::unique_ptr<nvinfer1::IExecutionContext> context_;
        std::shared_ptr<nvinfer1::ICudaEngine> engine_;
        std::weak_ptr<gotrt::GoBuffer> buffer_;
        std::weak_ptr<nvinfer1::Dims32> input_dim_;
        std::weak_ptr<std::vector<nvinfer1::Dims32>> output_dims_;

    };

} // gotrt

#endif //WINTRT_GO_ENGINE_H
