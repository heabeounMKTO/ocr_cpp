#include "yolo_model.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>

std::vector<Detection> run_inference(YoloModel *detector,
                                     cv::Mat *input_image) {

  // torch::Device device(torch::kCPU);
  std::vector<Detection> detections;
  torch::Tensor input_tensor;

  auto o_width = input_image->cols;
  auto o_height = input_image->rows;

  int new_w, new_h;
  float ratio;
  scale_wh(o_width, o_height, &new_w, &new_h, &ratio);
  fprintf(stdout, "new_w %d new_h %d\n", new_w, new_h);

  preprocess(input_image, &input_tensor, INFERENCE_SIZE);
  printf("input_image tensor shape: [");
  for (int i = 0; i < input_tensor.dim(); i++) {
    printf("%ld%s", input_tensor.size(i),
           i < input_tensor.dim() - 1 ? ", " : "");
  }
  printf("]\n");

  try {
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> _inputs;
    _inputs.push_back(input_tensor);

    auto output = detector->model.forward(_inputs).toTensor();

    output = output.to(torch::kCPU);

    /*  [1, 14, 8400]
     https://github.com/orgs/ultralytics/discussions/17254
     */
    int num_detections = output.size(1);
    int feats_size = output.size(2);
    int all_classes = num_detections - 5;
    auto nclasses = feats_size - 5;

    /* fprintf(stdout, "detections %d\n", num_detections);
    fprintf(stdout, "dims %ld\n", output.dim());
    printf("Output tensor shape: [");
    for (int i = 0; i < output.dim(); i++) {
        printf("%ld%s", output.size(i), i < output.dim()-1 ? ", " : "");
    }
    printf("]\n"); */

    for (int i = 0; i < num_detections; i++) {
      auto detection = output[0][i];
      float x = detection[0].item<float>();
      float y = detection[1].item<float>();
      float w = detection[2].item<float>();
      float h = detection[3].item<float>();
      fprintf(stdout, "RAW %f %f %f %f\n", x, y, w, h);

      int _x = (int) (((x - 0.5 * w) * ratio));
      int _y = (int) (((y - 0.5 * h) * ratio));
      int width = (int) ((w * ratio));
      int height = (int) ((h * ratio));

      fprintf(stdout, "UNSTRETCH %d %d %d %d\n", _x, _y, width, height);

      float obj_conf = detection[4].item<float>();
      printf("--------------\n");
      // fprintf(stdout, "confidence %f\n", obj_conf);
      // auto class_scores = detection.slice(0,5);
      //
      //
      //
      // for (int i = 0; i < num_detections; i++) {
      //   printf("\ncls scr %f\n", detection[i].item<float>());
      // }
      //
      // auto max_result = class_scores.max(0);
      // int class_id = std::get<1>(max_result).item<int>();
      // float class_conf = std::get<0>(max_result).item<float>();
      // fprintf(stdout, "final conf %f\nconf %f\n", class_conf * obj_conf,
      // obj_conf);
      printf("--------------\n");
    }
    return detections;
  } catch (const std::exception &e) {
    fprintf(stderr, "fucking torch forward error %s", e.what());
    throw;
  }
}

int main(int argc, char **argv) {
  YoloModel *keyinfo_model = create_yolo_detector(
      "/media/hbdesk/hb_desk_ext/yolov11cpp/models/id_ki_v1.2.pt.torchscript",
      "/media/hbdesk/hb_desk_ext/yolov11cpp/models/id_ki_classes.txt", 0.7, 0.7,
      640, 640);
  cv::Mat image = cv::imread("/media/hbdesk/hb_desk_ext/kh_id_keyinfo/kh_tg_id/472293344_122119262414603145_7291510996292981817_n.jpg");
  if (image.empty()) {
    fprintf(stderr, "Failed to load image\n");
    return -1;
  };

  std::vector<Detection> _a = run_inference(keyinfo_model, &image);
}
