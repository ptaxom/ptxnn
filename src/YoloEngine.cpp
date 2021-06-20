#include "YoloEngine.h"

namespace py = pybind11;
using NPImage = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

YoloEngine::YoloEngine(const char* model_name, const char* weight_path, int n_classes) :
    GeneralInferenceEngine(model_name, weight_path), n_classes_(n_classes)
{
    input_dim_ = bindings_explicit_dims_[0];
    channel_size_ = sizeof(dnnType) * input_dim_.d[2] * input_dim_.d[3];
    sample_size_ = channel_size_ * 3;
}

ListNPArray YoloEngine::predict(const std::vector<NPImage> &input)
{
    predict_async(input);
    return synchronize_async();
}


void YoloEngine::preprocess(int index_id, const NPImage &input)
{
    py::buffer_info input_buffer = input.request();
    uint8_t *input_ptr = static_cast<uint8_t*>(input_buffer.ptr);

    cv::Mat frame(input.shape(0), input.shape(1), CV_8UC3, input_ptr);
    cv::resize(frame, frame, cv::Size(input_dim_.d[3], input_dim_.d[2]));
    frame.convertTo(imagePreproc_, CV_32FC3, 1/255.0); 
    cv::split(imagePreproc_, bgr_);

    for(int channel = 0; channel < 3; channel++)
    {
        checkCuda(cudaMemcpyAsync(bindingsRT[0] + index_id * sample_size_ + channel * channel_size_, (void*)bgr_[channel].data, 
            channel_size_, cudaMemcpyHostToDevice, stream_));
    }
}

void YoloEngine::predict_async(const std::vector<NPImage> &input)
{
    mutex_.lock();
    batch_size_ = input.size();
    for(int batch_image = 0; batch_image < batch_size_; batch_image++)
        preprocess(batch_image, input[batch_image]);
}

ListNPArray YoloEngine::synchronize_async()
{

    mutex_.unlock();
}