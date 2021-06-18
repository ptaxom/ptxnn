#pragma once

#include "NvInfer.h"
#include <memory>
#include "NetworkRT.h"

struct TRTDeleter{
    
    template<class T>
    void operator()(T* obj) const {
        if (obj)
            obj->destroy();
    }
};

class GeneralInferenceEngine {
    // Static members, which would be shared between all engines
    static std::shared_ptr<nvinfer1::IBuilder> builderRT;
    static std::shared_ptr<nvinfer1::IBuilderConfig> configRT;
    static tk::dnn::PluginFactory *tkPlugins;
    static std::shared_ptr<nvinfer1::IRuntime> runtimeRT;

    // Class members, which defines engine
    std::shared_ptr<nvinfer1::INetworkDefinition> networkRT;
    nvinfer1::ICudaEngine *engineRT;
    nvinfer1::IExecutionContext *contextRT;

public:    
    GeneralInferenceEngine(const char* weight_path);
    virtual ~GeneralInferenceEngine() {};

    int getMaxBatchSize() {
        if(engineRT != nullptr)
            return engineRT->getMaxBatchSize();
        else
            return 0;
    }

    int getBuffersN() {
        if(engineRT != nullptr)
            return engineRT->getNbBindings();
        else 
            return 0;
    }

    void enqueue(int batchSize = 1);    

    void deserialize(const char *filename);
};
