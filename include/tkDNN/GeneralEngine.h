#pragma once

#include <memory>
#include <vector>
#include <iostream>
#include <map>
#include <errno.h>
#include <string.h> // memcpy
#include <stdlib.h>
#include <mutex>
#include <iostream>
#include <fstream>
#include <map>
#include <cuda.h>

#include "NvInfer.h"
#include "NetworkRT.h"
#include "utils.h"
#include "NvInferPlugin.h"

struct TRTDeleter{
    
    template<class T>
    void operator()(T* obj) const {
        if (obj)
            obj->destroy();
    }
};

size_t sample_size(nvinfer1::Dims dim);
size_t binding_size(nvinfer1::Dims dim);


class GeneralInferenceEngine {
    // Static members, which would be shared between all engines
    static std::shared_ptr<nvinfer1::IBuilder> builderRT;
    static std::shared_ptr<nvinfer1::IBuilderConfig> configRT;
    static tk::dnn::PluginFactory *tkPlugins;
    static std::shared_ptr<nvinfer1::IRuntime> runtimeRT;

    // Class members, which defines engine
    // ???
    std::shared_ptr<nvinfer1::INetworkDefinition> networkRT;
    // Deserialized CUDA engine, which perform model inference
    nvinfer1::ICudaEngine *engineRT;
    // Execution context of deserilized CUDA engine
    nvinfer1::IExecutionContext *contextRT;

    // Model name, which using for log messages
    std::string model_name_;
        
    // Vector of device pointers, which reffer to engine bindings
    std::vector<void*> bindingsRT;
    // CUDA Stream used for current engine
    cudaStream_t stream_;
    //
    std::vector<nvinfer1::Dims> bindings_dim_;
    // Host pointer, which is reserved for transmissions between host and input binding
    void* input_host_;
    // Vector of host pointers, which is reserved for transmissions between output bindings and host
    std::vector<void*> outputs_host_;

public:    
    GeneralInferenceEngine(const char* model_name, const char* weight_path);
    virtual ~GeneralInferenceEngine() {};

    void enqueue(int batchSize = 1);    

    void deserialize(const char *filename);
};
