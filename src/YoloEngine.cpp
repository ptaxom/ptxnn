#include "YoloEngine.h"
#include <fstream>

namespace py = pybind11;
using NPImage = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

// #FIXME: I actualy have no idea, about height and width order in nvinfer1::Dims, just relay on NCHW format
// For square input shape its not a problem, but..

YoloEngine::YoloEngine(const char* model_name, const char* weight_path, int n_classes, const float conf_threshold) :
    GeneralInferenceEngine(model_name, weight_path), n_classes_(n_classes), conf_threshold_(conf_threshold)
{
    input_dim_ = bindings_explicit_dims_[0];
    channel_size_ = sizeof(dnnType) * input_dim_.d[2] * input_dim_.d[3];
    sample_size_ = channel_size_ * 3;

    for(int i = 0; i < tkPlugins->n_yolos; i++) {
        auto *yRT = tkPlugins->yolos[i];
        int classes = yRT->classes;
        int num = yRT->num;
        int nMasks = yRT->n_masks;

        // make a yolo layer to interpret predictions
        tk::dnn::Yolo *yolo = new tk::dnn::Yolo(nullptr, classes, nMasks, ""); // yolo without input and bias
        yolo->mask_h = new dnnType[nMasks];
        yolo->bias_h = new dnnType[num * nMasks * 2];
        memcpy(yolo->mask_h, yRT->mask, sizeof(dnnType) * nMasks);
        memcpy(yolo->bias_h, yRT->bias, sizeof(dnnType) * num * nMasks * 2);
        yolo->input_dim = yolo->output_dim = tk::dnn::dataDim_t(1, yRT->c, yRT->h, yRT->w);
        yolo->classesNames = yRT->classesNames;
        yolo->nms_thresh = yRT->nms_thresh;
        yolo->nsm_kind = (tk::dnn::Yolo::nmsKind_t) yRT->nms_kind;
        yolo->new_coords = yRT->new_coords;
        heads.push_back(yolo);
    }
    dets = tk::dnn::Yolo::allocateDetections(tk::dnn::Yolo::MAX_DETECTIONS, n_classes_);
}

ListNPArray YoloEngine::predict_image(const std::vector<NPImage> &input)
{
    predict_image_async(input);
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
        checkCuda(cudaMemcpyAsync((char*)bindingsRT[0] + index_id * sample_size_ + channel * channel_size_, (void*)bgr_[2 - channel].data, 
            channel_size_, cudaMemcpyHostToDevice, stream_));
    }
}

void YoloEngine::predict_image_async(const std::vector<NPImage> &input)
{
    mutex_.lock();
    batch_size_ = input.size();
    for(int batch_image = 0; batch_image < batch_size_; batch_image++)
        preprocess(batch_image, input[batch_image]);
    contextRT->enqueue(engine_batch_size_, bindingsRT.data(), stream_, nullptr);

    for(int binding_id = 1; binding_id < engineRT->getNbBindings(); binding_id++)
    {
        checkCuda(cudaMemcpyAsync(outputs_host_[binding_id - 1], bindingsRT[binding_id], 
           binding_size(binding_id), cudaMemcpyDeviceToHost, stream_));
    }
}

inline float clip(float x)
{
    return std::max(0.f,  std::min(1.f, x));
}

NPArray YoloEngine::postprocess(int index_id)
{
    std::vector<dnnType*> rt_out;
 
    for(int i = 1; i < bindingsRT.size(); i++)
    {
        void* sample_ptr = (char*)bindingsRT[i] + index_id * (binding_size(i) / engine_batch_size_);
        rt_out.push_back(static_cast<dnnType*>(sample_ptr));
    }

    int nDets = 0;
    float H = input_dim_.d[2],
          W = input_dim_.d[3];
    for(int i = 0; i < heads.size(); i++) {
        heads[i]->dstData = rt_out[i];
        heads[i]->computeDetections(dets, nDets, W, H, conf_threshold_, heads[i]->new_coords);
    }
    tk::dnn::Yolo::mergeDetections(dets, nDets, n_classes_);

    std::vector<float> prediction;
    int n_actual_dets = 0;
    for(int j=0; j<nDets; j++) {
        tk::dnn::Yolo::box b = dets[j].bbox;
        float x0   = clip((b.x-b.w/2.) / W);
        float x1   = clip((b.x+b.w/2.) / W);
        float y0   = clip((b.y-b.h/2.) / H);
        float y1   = clip((b.y+b.h/2.) / H);
        
        int obj_class = -1;
        float prob = -1;
        for(int target_class_id = 0; target_class_id < n_classes_; target_class_id++)
            if (dets[j].prob[target_class_id] >= conf_threshold_)
                {
                    obj_class = target_class_id;
                    prob = dets[j].prob[target_class_id];
                    break;
                }

        if(obj_class >= 0) {
            for(float val: {x0, y0, x1, y1, (float)obj_class, prob})
                prediction.push_back(val);
            ++n_actual_dets;
        }
    }

    return NPArray({n_actual_dets, 6}, prediction.data());
}

ListNPArray YoloEngine::synchronize_async()
{
    checkCuda(cudaStreamSynchronize(stream_));
    std::vector<NPArray> sample_predictions;
    for(int batch_id = 0; batch_id < batch_size_; batch_id++)
        sample_predictions.push_back(postprocess(batch_id));
    mutex_.unlock();
    return sample_predictions;
}

PYBIND11_MODULE(_ptxnn, m) {
    py::class_<GeneralInferenceEngine>(m, "GeneralInferenceEngine")
            .def(py::init<const char*, const char*>())
            .def("predict", &GeneralInferenceEngine::predict)
            .def("predict_async", &GeneralInferenceEngine::predict_async)
            .def("batch_size", &GeneralInferenceEngine::batch_size)
            .def("np_input_shape", &GeneralInferenceEngine::np_input_shape)
            .def("synchronize_async", &GeneralInferenceEngine::synchronize_async);

    py::class_<YoloEngine, GeneralInferenceEngine>(m, "YoloEngine")
            .def(py::init<const char*, const char*, int, const float>())
            .def("predict_image", &YoloEngine::predict_image)
            .def("predict_image_async", &YoloEngine::predict_image_async)
            .def("synchronize_async", &YoloEngine::synchronize_async);

    m.def("set_severity", &set_severity, "Set TensorRT logger severity");
}