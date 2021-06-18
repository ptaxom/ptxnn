#include <iostream>
#include <map>
#include <errno.h>
#include <string.h> // memcpy
#include <stdlib.h>
#include <mutex>
#include <iostream>
#include <fstream>
#include <map>

#include "GeneralEngine.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"


using namespace nvinfer1;
using Severity = ILogger::Severity;

std::shared_ptr<nvinfer1::IBuilder> GeneralInferenceEngine::builderRT;
std::shared_ptr<nvinfer1::IBuilderConfig> GeneralInferenceEngine::configRT;
tk::dnn::PluginFactory *GeneralInferenceEngine::tkPlugins;
std::shared_ptr<nvinfer1::IRuntime> GeneralInferenceEngine::runtimeRT;


std::map<Severity, std::string> SEVERITY_COLORS = {
    {Severity::kINTERNAL_ERROR, "\033[91m\033[1m[CRITICAL]: "},
    {Severity::kERROR,                 "\033[91m[ERROR]:    "},
    {Severity::kWARNING,               "\033[93m[WARNING]:  "},
    {Severity::kINFO,                  "\033[92m[INFO]:     "},
    {Severity::kVERBOSE,               "\033[94m[DEBUG]:    "}
};

class EngineLogger : public ILogger {

    std::mutex log_guard;

public:
    void log(Severity severity, const char* msg) override {
        std::lock_guard<std::mutex> guard(log_guard);
        
        std::cout << SEVERITY_COLORS[severity] << msg << "\033[0m" <<  std::endl;
    }

} EngineLogger;

GeneralInferenceEngine::GeneralInferenceEngine(const char* weight_path)
{
    if (!GeneralInferenceEngine::builderRT)
    {
        std::shared_ptr<nvinfer1::IBuilder> ptr(createInferBuilder(EngineLogger), TRTDeleter());
        GeneralInferenceEngine::builderRT = ptr;
        std::shared_ptr<nvinfer1::IBuilderConfig> ptr2(ptr->createBuilderConfig(), TRTDeleter());
        GeneralInferenceEngine::configRT = ptr2;
        GeneralInferenceEngine::tkPlugins = new tk::dnn::PluginFactory();
        GeneralInferenceEngine::runtimeRT = std::shared_ptr<IRuntime>(createInferRuntime(EngineLogger), TRTDeleter());
        initLibNvInferPlugins(&EngineLogger, "");
    }
    deserialize(weight_path);
}

void GeneralInferenceEngine::deserialize(const char *filename) {

    char *gieModelStream{nullptr};
    size_t size{0};
    std::ifstream file(filename, std::ios::binary);
    std::stringstream info_string;

    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        gieModelStream = new char[size];
        file.read(gieModelStream, size);
        file.close();
    }
    else
    {
        info_string << "Couldn't open file at path " << filename;
        throw std::runtime_error(info_string.str());
    }

    info_string << "Loaded engine from " << filename << std::endl;
    EngineLogger.log(Severity::kINFO, info_string.str().c_str());
    engineRT = runtimeRT->deserializeCudaEngine(gieModelStream, size, (IPluginFactory *) GeneralInferenceEngine::tkPlugins);
}