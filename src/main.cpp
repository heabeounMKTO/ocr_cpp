#include "yolo_model.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>

int main(int argc, char **argv) {
  YoloModel *keyinfo_model = create_yolo_detector(
      "/media/hbdesk/hb_desk_ext/yolov11cpp/models/id_ki_v1.2.pt.torchscript",
      "/media/hbdesk/hb_desk_ext/yolov11cpp/models/id_ki_classes.txt",
      torch::kCPU, 0.3, 0.3, 640, 640);
  cv::Mat image =
      cv::imread("/media/hbdesk/hb_desk_ext/kh_id_keyinfo/kh_tg_id/b9ea8f60-2c5c-4cc1-a193-333874e78deb.jpeg");
  if (image.empty()) {
    fprintf(stderr, "Failed to load image\n");
    return -1;
  };
  std::cout << keyinfo_model->class_names << std::endl;
  // torch::Tensor input_tensor;
  // preprocess(&image, &input_tensor, INFERENCE_SIZE);
  std::vector<Detection> _a = run_inference(keyinfo_model, &image);
  for (int i = 0; i < _a.size(); i++ ) {
    fprintf(stdout, "conf %f\n", _a[i].confidence);
  std::cout << 
    "class_name " <<  keyinfo_model->class_names[_a[i].class_id] 
    << std::endl;

  }
}
