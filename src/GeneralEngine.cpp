#include "GeneralEngine.h"

using namespace nvinfer1;
using Severity = ILogger::Severity;

std::shared_ptr<nvinfer1::IBuilder> GeneralInferenceEngine::builderRT;
std::shared_ptr<nvinfer1::IBuilderConfig> GeneralInferenceEngine::configRT;
std::shared_ptr<nvinfer1::IRuntime> GeneralInferenceEngine::runtimeRT;


std::map<Severity, std::string> SEVERITY_COLORS = {
    {Severity::kINTERNAL_ERROR, "\033[91m\033[1m[CRITICAL]: "},
    {Severity::kERROR,                 "\033[91m[ERROR]:    "},
    {Severity::kWARNING,               "\033[93m[WARNING]:  "},
    {Severity::kINFO,                  "\033[92m[INFO]:     "},
    {Severity::kVERBOSE,               "\033[94m[DEBUG]:    "}
};

std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& obj)
{
    for(int i = 0; i < obj.nbDims - 1; i++)
        os << obj.d[i] << "x";
    os << obj.d[obj.nbDims - 1];
    return os;
}

class EngineLogger : public ILogger {

    std::mutex log_guard;

public:

    Severity logger_severity = Severity::kWARNING;

    void log(Severity severity, const char* msg) override {
        std::lock_guard<std::mutex> guard(log_guard);
        
        if (severity <= logger_severity)
            std::cout << SEVERITY_COLORS[severity] << msg << "\033[0m" <<  std::endl;
    }

    template <class T>
    void log(Severity severity, const char* msg, T model_name){
        std::stringstream message_ss;
        message_ss << msg << " " << model_name;
        log(severity, message_ss.str().c_str());
    }

} EngineLogger;

void set_severity(int severity)
{
    if (severity < 0 || severity > 4)
        throw std::runtime_error("Unsupported severity");
    EngineLogger.logger_severity = (Severity)severity;
}

GeneralInferenceEngine::GeneralInferenceEngine(const char* model_name, const char* weight_path): model_name_(model_name)
{
    if (!GeneralInferenceEngine::builderRT)
    {
        cudaSetDevice(0);
        std::shared_ptr<nvinfer1::IBuilder> ptr(createInferBuilder(EngineLogger), TRTDeleter());
        GeneralInferenceEngine::builderRT = ptr;
        std::shared_ptr<nvinfer1::IBuilderConfig> ptr2(ptr->createBuilderConfig(), TRTDeleter());
        GeneralInferenceEngine::configRT = ptr2;
        GeneralInferenceEngine::runtimeRT = std::shared_ptr<IRuntime>(createInferRuntime(EngineLogger), TRTDeleter());
        initLibNvInferPlugins(&EngineLogger, "");
    }
    tkPlugins = new tk::dnn::PluginFactory();
    deserialize(weight_path);
    
    engine_batch_size_ = engineRT->getMaxBatchSize();
    // if compiled from onnx file with explicit dim and mismatched batchsize, than smthg goes wrong
    if (!engineRT->hasImplicitBatchDimension() && engine_batch_size_ != engineRT->getBindingDimensions(0).d[0])
        throw std::runtime_error("Mismatched explicit batchsize and engine batchsize");
    EngineLogger.log(Severity::kINFO, "Used batchsize of", engine_batch_size_);
    
    contextRT = engineRT->createExecutionContext();
    EngineLogger.log(Severity::kINFO, "Created execution context for", model_name_);
    

    int n_inputs = 0;
    for(int i = 0; i < engineRT->getNbBindings(); i++)
    {
        void *device_ptr;
        Dims dim = engineRT->getBindingDimensions(i);
        size_t current_binding_size = binding_size(dim);
        checkCuda(cudaMalloc(&device_ptr, current_binding_size));
        
        Dims explicit_dim;
        if (engineRT->hasImplicitBatchDimension())
        {
            explicit_dim.nbDims = dim.nbDims + 1;
            explicit_dim.d[0] = engine_batch_size_;
            for(int j = 0; j < dim.nbDims; j++)
                explicit_dim.d[1 + j] = dim.d[j];
        }
        else
        {
            explicit_dim = dim;
            explicit_dim.d[0] = engine_batch_size_;
        }
        std::vector<int> shape;
        for(int j = 0; j < explicit_dim.nbDims; j++)
            shape.push_back(explicit_dim.d[j]);

        if (engineRT->bindingIsInput(i))
        {
            ++n_inputs;
            bindingsRT.insert(bindingsRT.begin(), device_ptr);
            bindings_dim_.insert(bindings_dim_.begin(), dim);
            bindings_explicit_dims_.insert(bindings_explicit_dims_.begin(), explicit_dim);
            numpy_shapes_.insert(numpy_shapes_.begin(), shape);
        }
        else
        {
            bindingsRT.push_back(device_ptr);
            bindings_dim_.push_back(dim);
            bindings_explicit_dims_.push_back(explicit_dim);
            numpy_shapes_.push_back(shape);
        }
    
        std::stringstream message_str;
        message_str << "Allocated " << current_binding_size << " bytes for binding " << engineRT->getBindingName(i);
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
    engineRT = runtimeRT->deserializeCudaEngine(gieModelStream, size, (IPluginFactory *) tkPlugins);
}

void GeneralInferenceEngine::enqueue(){
    checkCuda(cudaMemcpyAsync(bindingsRT[0], input_host_, 
        binding_size(0), cudaMemcpyHostToDevice, stream_));
    
    contextRT->enqueue(engine_batch_size_, bindingsRT.data(), stream_, nullptr);

    for(int binding_id = 1; binding_id < engineRT->getNbBindings(); binding_id++)
    {
        checkCuda(cudaMemcpyAsync(outputs_host_[binding_id - 1], bindingsRT[binding_id], 
           binding_size(binding_id), cudaMemcpyDeviceToHost, stream_));
    }
}

void GeneralInferenceEngine::synchronize(){
    checkCuda(cudaStreamSynchronize(stream_));
}

size_t GeneralInferenceEngine::binding_size(int index)
{
    return binding_size(bindings_dim_[index]);
}

size_t GeneralInferenceEngine::binding_size(Dims dim)
{
    size_t size = sizeof(dnnType);
    if (engineRT->hasImplicitBatchDimension())
        size *= engine_batch_size_;

    for(int i = 0; i < dim.nbDims; i++)
        size *= dim.d[i];
    return size;
}

ListNPArray GeneralInferenceEngine::predict(const NPArray &input)
{
    predict_async(input);
    return synchronize_async();
}

void GeneralInferenceEngine::predict_async(const NPArray &input)
{
    mutex_.lock();
    is_inferencing_ = true;
    bool has_valid_size = input.ndim() == bindings_explicit_dims_[0].nbDims;
    for(int i = 0; i < input.ndim() && has_valid_size; i++)
        has_valid_size &= bindings_explicit_dims_[0].d[i] == input.shape(i);
    if (!has_valid_size)
    {
        std::stringstream ss;
        ss << "Invalid input numpy array. Expected numpy with shape " << bindings_explicit_dims_[0];
        is_inferencing_ = false;
        mutex_.unlock();
        throw std::runtime_error(ss.str().c_str());
    }
    py::buffer_info input_buffer = input.request();
    dnnType *input_ptr = static_cast<dnnType*>(input_buffer.ptr);
    
    checkCuda(cudaMemcpyAsync(bindingsRT[0], input_ptr, 
        binding_size(0), cudaMemcpyHostToDevice, stream_));
    
    contextRT->enqueue(engine_batch_size_, bindingsRT.data(), stream_, nullptr);

    for(int binding_id = 1; binding_id < engineRT->getNbBindings(); binding_id++)
    {
        checkCuda(cudaMemcpyAsync(outputs_host_[binding_id - 1], bindingsRT[binding_id], 
           binding_size(binding_id), cudaMemcpyDeviceToHost, stream_));
    }
}

void on_execution_end_gie(void *data)
{
    GeneralInferenceEngine *engine = static_cast<GeneralInferenceEngine*>(data);
    engine->is_inferencing_ = false;
}

void GeneralInferenceEngine::predict_callbacked(const NPArray &input)
{
    predict_async(input);
    checkCuda(cudaLaunchHostFunc(stream_, &on_execution_end_gie, this));
}

ListNPArray GeneralInferenceEngine::synchronize_callback()
{
    ListNPArray predictions;
    for(int i = 1; i < bindingsRT.size(); i++)
    {
        predictions.push_back(NPArray(numpy_shapes_[i], (dnnType*)outputs_host_[i - 1]));
    }
    mutex_.unlock();
    return predictions;
}

ListNPArray GeneralInferenceEngine::synchronize_async()
{
    checkCuda(cudaStreamSynchronize(stream_));
    ListNPArray predictions;
    for(int i = 1; i < bindingsRT.size(); i++)
    {
        predictions.push_back(NPArray(numpy_shapes_[i], (dnnType*)outputs_host_[i - 1]));
    }
    is_inferencing_ = false;
    mutex_.unlock();
    return predictions;
}

size_t GeneralInferenceEngine::batch_size() const
{
    return engine_batch_size_;
}

bool GeneralInferenceEngine::is_inferencing() const
{
    return is_inferencing_;
}

py::tuple GeneralInferenceEngine::np_input_shape() const
{
    auto shape = py::tuple(numpy_shapes_[0].size());
    for(int i = 0; i < numpy_shapes_[0].size(); i++)
        shape[i] = numpy_shapes_[0][i];
    return shape;
}

void convert_yolo(const char* cfg_path, const char* layers_folder, const char* names_path, const char* engine_name, char* mode, int batchsize)
{
    tk::dnn::Network *net = tk::dnn::darknetParser(cfg_path, layers_folder, names_path, mode, batchsize);
    tk::dnn::NetworkRT *netRT = new tk::dnn::NetworkRT(net, net->getNetworkRTName(engine_name));
    delete netRT;
    delete net;
}