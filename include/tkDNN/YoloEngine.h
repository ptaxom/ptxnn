#pragma once

#include "GeneralEngine.h"
#include "utils.h"

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <mutex>
#include "opencv2/opencv.hpp"

namespace py = pybind11;
using NPImage = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

class YoloEngine : public GeneralInferenceEngine
{
    int batch_size_;
    int n_classes_;
    cv::Mat bgr_[3];
    cv::Mat imagePreproc_;
    size_t channel_size_;
    size_t sample_size_;
    nvinfer1::Dims input_dim_;
    std::vector<tk::dnn::Yolo*> heads;
    tk::dnn::Yolo::detection *dets = nullptr;
    float conf_threshold_;

    void preprocess(int index_id, const NPImage &input);
    NPArray postprocess(int index_id);

public:
    YoloEngine(const char* model_name, const char* weight_path, int n_classes, const float conf_threshold);

    ListNPArray predict_image(const std::vector<NPImage> &input);
    void predict_image_async(const std::vector<NPImage> &input);
    ListNPArray synchronize_async();

};