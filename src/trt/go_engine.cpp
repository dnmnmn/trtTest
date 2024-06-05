//
// Created by gopizza on 2024-06-04.
//

#include "go_engine.h"
#include <iostream>
#include "../../common/logger.h"
#include "../../common/common.h"

using namespace gotrt;

void GoEngine::Initialize() {
    std::cout << "GoEngine::initialize()" << std::endl;
}

void GoEngine::Release() {
    std::cout << "GoEngine::release()" << std::endl;
}

bool GoEngine::LoadEngine(ModelParams _model_params,
                          std::shared_ptr<nvinfer1::Dims32> _input_dim,
                          std::shared_ptr<std::vector<nvinfer1::Dims32>> _output_dims)
{
    std::cout << "GoEngine::load_engine()" << std::endl;

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger));
    if (!builder) return false;

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) return false;

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return false;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger));
    if (!parser) return false;

    auto constructed = ConstructNetwork(builder, network, config, parser, _model_params);
    if (!constructed) return false;

    stream_ = std::make_unique<cudaStream_t>();
    if (cudaStreamCreateWithFlags(stream_.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        stream_.reset(nullptr);
        return false;
    }
    config->setProfileStream(*stream_);

    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) return false;

    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime) return false;

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if(!engine_) {
        engine_.reset();
        return false;
    }
    if(engine_->getNbBindings() < 1)
        std::cout << "error " << engine_->getNbBindings() << std::endl;
    *_input_dim = engine_->getBindingDimensions(0);
    _output_dims->push_back(engine_->getBindingDimensions(engine_->getNbBindings() - 1));
    for (int i = 1; i < engine_->getNbBindings() - 1; i++) {
        _output_dims->push_back(engine_->getBindingDimensions(i));
    }
    this->input_dim_ = _input_dim;
    this->output_dims_ = _output_dims;
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if(!context_) return false;
    return true;
}

bool GoEngine::ConstructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
                                std::unique_ptr<nvinfer1::INetworkDefinition>& network,
                                std::unique_ptr<nvinfer1::IBuilderConfig>& config,
                                std::unique_ptr<nvonnxparser::IParser>& parser,
                                ModelParams model_params)
{
    auto parsed = parser->parseFromFile(model_params.wight_path.c_str(),
                                        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    if (model_params.fp16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (model_params.int8)
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    // samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

bool GoEngine::Inference(void* _tensor)
{
    std::cout << "GoEngine::inference()" << std::endl;
    return context_->executeV2(&_tensor);
}