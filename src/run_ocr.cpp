#include "../include/onnx_utils.h"
#include "yolo_model.h"
#include <cstdint>
#include <cstdio>
#include <onnxruntime_c_api.h>
#include <opencv2/opencv.hpp>
#include <ostream>

int main (int argc, char* argv[]) {
    OnnxModel* loaded_model = onnx_model_load(
    "/media/hbdesk/hb_desk_ext/ocr_core/models/id_ki_v1.1.onnx");
    OnnxModelInfo info = onnx_model_get_info(loaded_model);
    cv::Mat _aa = cv::imread ("/home/hbdesk/Pictures/vn_passport/test1.jpg");
    cv::Mat _load_image         = letterbox(&_aa);
    std::vector<std::string> _a = get_class_names ("./res.txt");
    cv::imwrite ("./test_1_lb.png", _load_image);
    std::vector<int64_t> input_tensor_shape = { 1, 3, 640, 640 };
    OrtValue* input_tensor                  = NULL;
    mat_to_onnx_value (_load_image, &info, loaded_model, input_tensor);
    OrtValue* output_tensor = NULL;
    
    printf ("[DEBUG] load output shit : %s\ninput : %s\n", &info.input_names, &info.output_names);
    // loaded_model->api->Run(
    //   loaded_model->session,
    //   NULL,
    //   info.input_names,
    //   input_tensor,
    //   1,
    //   info.output_names,
    //   1,
    //   output_tensor
    // );
    return 0;
}
