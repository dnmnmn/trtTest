/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "common/argsParser.h"
#include "common/buffers.h"
#include "common/common.h"
#include "common/logger.h"
#include "common/parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <cstdio>
#include <cstdint>

#include "opencv2/opencv.hpp"
#include <chrono>
#include "src/postprocess.h"
#include "src/preprocess.h"
#include "src/stream/go_camera.h"
#include "src/Timer.h"
#include "src/trt/GoBuffer.h"
#include "src/stream/go_camera.h"
#include "src/stream/camera_stream.h"
#include "src/stream/video_stream.h"
#include "src/stream/image_stream.h"
#include "src/aistt.h"

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.sample_onnx_mnist";

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!

class SampleOnnxMNIST
{
public:
    SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
            : mParams(params)
            , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();
    bool release();
private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    std::shared_ptr<nvinfer1::Dims32> mInputDims;  //!< The dimensions of the input to the network.
    std::shared_ptr<std::vector<nvinfer1::Dims>> mOutputDims;  //!< The dimensions of the output to the network.
    std::vector<nvinfer1::Dims32> mDims32;  //!< The dimensions of the output to the network.
//    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
//    nvinfer1::Dims mOutputDims1; //!< The dimensions of the output to the network.
//    nvinfer1::Dims mOutputDims2; //!< The dimensions of the output to the network.
//    nvinfer1::Dims mOutputDims3; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    std::shared_ptr<InputStream> camera = nullptr;
    std::shared_ptr<PreProcess> preprocess = nullptr;
    std::shared_ptr<PostProcess> postprocess = nullptr;
    std::shared_ptr<gotrt::GoBuffer> gobuffer = nullptr;
    Timer timer;

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                          SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers, const cv::Mat image);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers, cv::Mat image);
    // bool Tensor_Debug(const samplesCommon::BufferManager& buffers);
    bool Tensor_Debug(float* tensor, int32_t* dims, int32_t size);
};

bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleOnnxMNIST::release()
{
    if(camera!=nullptr) camera->Release();
    preprocess->release();
    postprocess->release();
    if(gobuffer!=nullptr) gobuffer->release();
    return true;
}
//! \return
bool SampleOnnxMNIST::build()
{

//    auto image = camera.get_frame();
//    while(true)
//    {
//        image = camera.get_frame().clone();
//        cv::imshow("image", image);
//        auto stop = cv::waitKey(100);
//        if (stop == 'q' || stop == 'Q')
//        {
//            break;
//        }
//    }
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
            = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    //ASSERT(network->getNbInputs() == 1);
    mInputDims = std::make_shared<nvinfer1::Dims32>(network->getInput(0)->getDimensions());
    // mInputDims = network->getInput(0)->getDimensions();
    //ASSERT(mInputDims.nbDims == 4);

    //ASSERT(network->getNbOutputs() == 1);
    mOutputDims = std::make_shared<std::vector<nvinfer1::Dims>>();
    mOutputDims->push_back(network->getOutput(0)->getDimensions());
//    mOutputDims1 = network->getOutput(1)->getDimensions();
//    mOutputDims2 = network->getOutput(2)->getDimensions();
//    mOutputDims3 = network->getOutput(3)->getDimensions();
    // ASSERT(mOutputDims.nbDims == 2);
    //ASSERT(mOutputDims.nbDims == 3);
    int org_image_width = 4032;
    int org_image_height = 1504;
    std::string file_name = "test.jpg";
    if(ends_with(file_name, ".jpg") || ends_with(file_name, ".png"))
    {
        camera = std::make_shared<ImageStream>();
    }
    else if(ends_with(file_name, ".mp4") || ends_with(file_name, ".avi"))
    {
        camera = std::make_shared<VideoStream>();
    }
    else
    {
        camera = std::make_shared<CameraStream>();
    }
    camera->Initialize(file_name, org_image_height, org_image_width);

    //camera.initialize();
    //camera.camera_load(org_image_height, org_image_width);
    bool is_obb = false;
    if (is_obb)
        postprocess = std::make_shared<PostProcessOBB>();
    else
        postprocess = std::make_shared<PostProcessBase>();
    postprocess->initialize(mInputDims, mOutputDims);
    preprocess = std::make_shared<PreProcess>();
    preprocess->initialize(mInputDims, mOutputDims, org_image_height, org_image_width);


    for(int i = 0;i < mEngine->getNbBindings(); i++)
    {
        mDims32.push_back(mEngine->getBindingDimensions(i));
        printf("binding: %d, dims: ", i);
        for(int j = 0; j < mDims32[i].nbDims; j++) printf("%d, ", mDims32[i].d[j]);
        printf("\n");
    }
    gobuffer = std::make_shared<gotrt::GoBuffer>();
    gobuffer->initialize(1, mInputDims, mOutputDims);
    gobuffer->ready();
    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                                        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                                        SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
                                        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    // samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

bool SampleOnnxMNIST::Tensor_Debug(float* tensor, int32_t* dims, int32_t size)
{
    //float* output = new float[dims[0] * dims[1]  * dims[2] * dims[3]];
    //memset(output, 0, dims[1]  * dims[2] * dims[3] * sizeof(float));
    //memcpy(output, tensor, dims[1]  * dims[2] * dims[3] * sizeof(float));

    // Align
    int multiple = sqrt(dims[1]) + 1;
    int image_width = dims[3] * multiple;
    int image_height = dims[2] * multiple;

    float* output1_align = new float[image_height * image_width];
    memset(output1_align, 0.5, image_height * image_width * sizeof(float));


    namedWindow( "output1", cv::WINDOW_AUTOSIZE);
    for(int i = 0; i < dims[1]; i++)
    {
        int img_x = i % multiple;
        int img_y = i / multiple;
        int idx_x = img_x * dims[3];
        int idx_y = img_y * dims[2] * image_width;
        for(int j = 0; j < dims[2]; j++)       // height
        {
            for(int k = 0; k < dims[3]; k++) // width
            {
                output1_align[idx_y + idx_x + image_width * j + k] = tensor[i * dims[2] * dims[3] + j * dims[3] + k];
            }
        }

    }
    // tensor to 2d image
    cv::Mat output_image1(image_height, image_width, CV_32FC1, output1_align);
    cv::resize(output_image1, output_image1, cv::Size(image_width * size, image_height * size), 0, 0, cv::INTER_NEAREST);
    cv::Mat color_image;
    // CV_32FC1 to CV_8UC1
    output_image1.convertTo(output_image1, CV_8UC1, 255.0);
    cv::applyColorMap(output_image1, color_image, cv::COLORMAP_JET);
    cv::imshow("output1", color_image);
    cv::waitKey(0);
    cv::imwrite("output" + std::to_string(size) + ".jpg", color_image);
    delete [] output1_align;
    //delete [] output;
    return true;
}
//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer()
{
    // Create RAII buffer manager object
    timer.start();
    samplesCommon::BufferManager buffers(mEngine);


    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    cv::Mat image = camera->GetFrame().clone();
    timer.start();
    if (!processInput(buffers, image))
    {
        return false;
    }
    auto infer_time = timer.end();
    sample::gLogInfo << "PreProcess time: " << infer_time << "ms" << std::endl;
    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();
    timer.start();
    // bool status = context->executeV2(buffers.getDeviceBindings().data());
    bool status = context->executeV2(gobuffer->getDeviceBindings().data());
    infer_time = timer.end();
    sample::gLogInfo << "Inference time: " << infer_time << "ms" << std::endl;
    //
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();
    gobuffer->cpyOutputToHost(0);
    // Verify results
    timer.start();
    if (!verifyOutput(buffers, image))
    {
        return false;
    }
    infer_time = timer.end();
    sample::gLogInfo << "PostProcess time: " << infer_time << "ms" << std::endl;
//    float* tensor1 = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));
//    if(!Tensor_Debug(tensor1, mOutputDims1.d, 1)) return false;
//    float* tensor2 = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[2]));
//    if(!Tensor_Debug(tensor2, mOutputDims2.d, 1)) return false;
//    float* tensor3 = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[3]));
//    if(!Tensor_Debug(tensor3, mOutputDims3.d, 2)) return false;

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers, cv::Mat image)
{
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    void* preprocess_data = preprocess->opencv_preprocess(&image);

    memcpy(hostDataBuffer, preprocess_data, mInputDims->d[2] * mInputDims->d[3] * 3 * sizeof(float));
    memcpy(gobuffer->getInputTensor(),preprocess_data, mInputDims->d[2] * mInputDims->d[3] * 3 * sizeof(float));
    gobuffer->cpyInputToDevice();
    return true;
}


//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!

bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers, cv::Mat image)
{
    // float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float* output = gobuffer->getOutputTensor(0);
    // postprocess->debug(output);
    postprocess->clear();
    postprocess->normalize(output, 0.25);
    auto normalize_box= postprocess->weighted_boxes_fusion();
    std::vector<std::vector<float>> box;
    postprocess->denormalize(normalize_box, 100, &box);

    cv::Mat reimage;

    cv::resize(image, reimage, cv::Size(mInputDims->d[3], mInputDims->d[2])); // cv::Size(width, height)
    std::vector<cv::Mat> crop_images;
    for(int i = 0; i < box.size(); i++)
    {
//        cv::line(reimage, cv::Point(box[i][0], box[i][1]), cv::Point(box[i][2], box[i][3]), cv::Scalar(0, 255, 0), 2);
//        cv::line(reimage, cv::Point(box[i][2], box[i][3]), cv::Point(box[i][4], box[i][5]), cv::Scalar(0, 255, 0), 2);
//        cv::line(reimage, cv::Point(box[i][4], box[i][5]), cv::Point(box[i][6], box[i][7]), cv::Scalar(0, 255, 0), 2);
//        cv::line(reimage, cv::Point(box[i][6], box[i][7]), cv::Point(box[i][0], box[i][1]), cv::Scalar(0, 255, 0), 2);
        cv::rectangle(reimage, cv::Point(box[i][0], box[i][1]), cv::Point(box[i][2], box[i][3]), cv::Scalar(0, 255, 0), 2);
        // cv::rectangle(reimage, cv::Point(normalize_box[i][0] * mInputDims.d[2], normalize_box[i][1] * mInputDims.d[2]), cv::Point(normalize_box[i][2] * mInputDims.d[2], normalize_box[i][3] * mInputDims.d[2]), cv::Scalar(0, 0, 255), 2);
        //cv::Mat crop(640, 640, CV_32FC3, preprocess->crop(box[i][0], box[i][1], box[i][2], box[i][3]));
        //crop_images.push_back(crop);
    }

    cv::imshow("image", reimage);
    if(cv::waitKey(1) == 'q')
        return false;
    return true;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("weights/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "best.onnx";
    params.inputTensorNames.push_back("images");
    params.outputTensorNames.push_back("output0");
    //params.outputTensorNames.push_back("onnx::Reshape_421");    // [1, 65, 52, 52]
//    params.outputTensorNames.push_back("654");    // [1, 65, 26, 26]
//    params.outputTensorNames.push_back("onnx::Reshape_733");    // [1, 65, 13, 13]

    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
            << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
            << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    // args.batch = 2;
    // args.runInInt8 = true;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleOnnxMNIST sample(initializeSampleParams(args));
    AISTT aistt = AISTT();
    aistt.Initialize();

    aistt.CreateModule(1504, 4032, false, false);

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;



    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    while(true)
    {

        if (!sample.infer())
        {
            break;
        }
    }
    if (!sample.release())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    aistt.Release();
    return sample::gLogger.reportPass(sampleTest);
}

