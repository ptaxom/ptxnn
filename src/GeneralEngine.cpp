#include "GeneralEngine.h"

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


size_t sample_size(nvinfer1::Dims dim)
{
    size_t size = 1;
    for(int i = 1; i < dim.nbDims; i++)
        size *= dim.d[i];
    return size;
}

size_t binding_size(nvinfer1::Dims dim)
{
    size_t size = 1;
    for(int i = 0; i < dim.nbDims; i++)
        size *= dim.d[i];
    return size;
}

class EngineLogger : public ILogger {

    std::mutex log_guard;

public:
    void log(Severity severity, const char* msg) override {
        std::lock_guard<std::mutex> guard(log_guard);
        
        std::cout << SEVERITY_COLORS[severity] << msg << "\033[0m" <<  std::endl;
    }

    template <class T>
    void log(Severity severity, const char* msg, T model_name){
        std::stringstream message_ss;
        message_ss << msg << " " << model_name;
        log(severity, message_ss.str().c_str());
    }

} EngineLogger;

GeneralInferenceEngine::GeneralInferenceEngine(const char* model_name, const char* weight_path): model_name_(model_name)
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
    
    contextRT = engineRT->createExecutionContext();
    EngineLogger.log(Severity::kINFO, "Created execution context for", model_name_);

    int n_inputs = 0;
    for(int i = 0; i < engineRT->getNbBindings(); i++)
    {
        void *device_ptr;
        Dims dim = engineRT->getBindingDimensions(i);
        size_t binding_size = engineRT->getMaxBatchSize() * sample_size(dim) * sizeof(dnnType);
        checkCuda(cudaMalloc(&device_ptr, binding_size));

        if (engineRT->bindingIsInput(i))
        {
            ++n_inputs;
            bindingsRT.insert(bindingsRT.begin(), device_ptr);
            bindings_dim_.insert(bindings_dim_.begin(), engineRT->getBindingDimensions(i));
        }
        else
        {
            bindingsRT.push_back(device_ptr);
            bindings_dim_.push_back(engineRT->getBindingDimensions(i));
        }
    
        std::stringstream message_str;
        message_str << "Allocated " << binding_size << " bytes for binding " << engineRT->getBindingName(i);
        EngineLogger.log(Severity::kVERBOSE, message_str.str().c_str());
    }
    
    if (n_inputs != 1)
    {
        std::stringstream info_string;
        info_string << "Number of inputs(" << n_inputs << ") is not supported currently";
        throw std::runtime_error(info_string.str());
    }

    checkCuda(cudaStreamCreate(&stream_));
    EngineLogger.log(Severity::kINFO, "Created CUDA stream for", model_name_);

    checkCuda(cudaMallocHost(&input_host_, binding_size(bindings_dim_[0])));
    for(int binding_id = 1; binding_id < engineRT->getNbBindings(); binding_id++)
    {
        void *host_ptr;
        checkCuda(cudaMallocHost(&host_ptr, binding_size(bindings_dim_[binding_id])));
        outputs_host_.push_back(host_ptr);
    }
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

    
    EngineLogger.log(Severity::kINFO, "Loaded engine from", filename);
    engineRT = runtimeRT->deserializeCudaEngine(gieModelStream, size, (IPluginFactory *) GeneralInferenceEngine::tkPlugins);
}