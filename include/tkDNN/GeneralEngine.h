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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

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



std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& obj);

namespace py = pybind11;
using NPArray = py::array_t<dnnType, py::array::c_style | py::array::forcecast>;
using ListNPArray = std::vector<NPArray>;

class GeneralInferenceEngine {
protected:
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
    // Batchsize with engine was compiled
    int engine_batch_size_ = -1;
    // Vector with shapes of bindings
    std::vector<nvinfer1::Dims> bindings_dim_;
    // Host pointer, which is reserved for transmissions between host and input binding
    void* input_host_ = nullptr;
    // Vector of host pointers, which is reserved for transmissions between output bindings and host
    std::vector<void*> outputs_host_;
    // Vector of explicit dims of bindings
    std::vector<nvinfer1::Dims> bindings_explicit_dims_;
    // Vector of numpy shapes
    std::vector<std::vector<int>> numpy_shapes_;

public:    
    GeneralInferenceEngine(const char* model_name, const char* weight_path);
    virtual ~GeneralInferenceEngine() {};

    // Deserialization of engine file from filepath
    void deserialize(const char *filename);
    // Enqueue pipeline in stream
    void enqueue();
    // Wait for execution
    void synchronize();
    // Get size of binding by ID
    size_t binding_size(int index);
    // Get size of dimenstions
    size_t binding_size(nvinfer1::Dims dim);

    ListNPArray predict(const NPArray &input);
    void predict_async(const NPArray &input);
    ListNPArray synchronize_async();
};