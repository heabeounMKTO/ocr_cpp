#include "../include/onnx_utils.h"
#include <cstdio>
#include "yolo_model.h"
#include <opencv2/opencv.hpp>
#include <ostream>


int main(int argc, char* argv[]) {
  OnnxModel *init_ok  = onnx_model_load("/media/hbdesk/hb_desk_ext/ocr_core/models/id_ki_v1.1.onnx");
  OnnxModelInfo info = onnx_model_get_info(init_ok);
  cv::Mat _aa = cv::imread("/home/hbdesk/Pictures/vn_passport/test1.jpg");
  cv::Mat _load_image = letterbox(&_aa);
  std::vector<std::string> _a = get_class_names("./res.txt");
  return 0;
}
