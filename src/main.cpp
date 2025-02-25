#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>

void preprocess(cv::Mat *image, torch::Tensor *output, int target_size) {
  cv::Mat resized;
  cv::resize(*image, resized, cv::Size(target_size, target_size));
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  cv::Mat float_img;
  resized.convertTo(float_img, CV_32F, 1.0 / 255.0);
  *output = torch::from_blob(float_img.data, {1, target_size, target_size, 3},
                             torch::kFloat32);
  *output = output->permute({0, 3, 1, 2});
}

float compute_iou(cv::Rect *box1, cv::Rect *box2) {
  int x1 = std::max(box1->x, box2->x);
  int y1 = std::max(box1->y, box2->y);
  int x2 = std::min(box1->x + box1->width, box2->x + box2->width);
  int y2 = std::min(box1->y + box1->height, box2->y + box2->height);

  if (x2 <= x1 || y2 <= y1) {
    return 0.0f;
  }
  float intersection = (x2 - x1) * (y2 - y1);
  float area1 = box1->width * box1->height;
  float area2 = box2->width * box2->height;
  float union_area = area1 + area2 - intersection;
  return intersection / union_area;
}

typedef struct {
  cv::Rect box;
  float confidence;
  int class_id;
} Detection;

typedef struct YoloModel {
  torch::jit::script::Module model;
  std::vector<std::string> class_names;
  float conf_thresh;
  float nms_thresh;
  int input_w;
  int input_h;
} YoloModel;

YoloModel *create_yolo_detector(const char *model_path,
                                const char *classes_path, float conf_thresh,
                                float nms_thresh, int w, int h) {
  YoloModel *detector = new YoloModel();
  try {
    detector->model = torch::jit::load(model_path);
    detector->model.eval();
    std::ifstream class_file(classes_path);
    if (!class_file.is_open()) {
        printf("Failed to open classes file\n");
        delete detector;
        return NULL;
    }

    std::string line;
    while (std::getline(class_file, line)) {
      detector->class_names.push_back(line);
    }

  } catch (const std::exception &e) {
    fprintf(stderr, "Error creating keypoints detector %s", e.what());
    delete detector;
    return NULL;
  }

  detector->conf_thresh = conf_thresh;
  detector->nms_thresh = nms_thresh;
  detector->input_w = w;
  detector->input_h = h;
  return detector;
}

std::vector<Detection> run_inference(YoloModel *detector,
                                     cv::Mat *input_image) {

  std::vector<Detection> detections;
  torch::Tensor input_tensor;
  preprocess(input_image, &input_tensor, 640);

  try {

    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> _inputs;
    _inputs.push_back(input_tensor);

    auto output = detector->model.forward(_inputs).toTensor();
    // output = output.to(torch::kCPU);
     
    int num_detections = output.size(2);
    fprintf(stdout, "detections %d\n", num_detections);
    fprintf(stdout, "dims %ld\n", output.dim()); 

    /* printf("Output tensor shape: [");
    for (int i = 0; i < output.dim(); i++) {
        printf("%ld%s", output.size(i), i < output.dim()-1 ? ", " : "");
    }
    printf("]\n"); */



    for (int i = 0; i < num_detections; i++) {
      // float confidence = output[0][i][3].item<float>();
      // fprintf(stdout, "confidence %f\n", confidence);
      // fprintf(stdout, "confidence %f", confidence);
      // if (confidence >= detector->conf_thresh) {
      //   fprintf(stdout, "confidence %f\n", confidence);
      // }
    }
    return detections;
  } catch (const std::exception &e) {
    fprintf(stderr, "fucking torch error %s", e.what());
    throw;
  }
}

int main(int argc, char **argv) {
  YoloModel *keyinfo_model = create_yolo_detector(
      "/media/hbdesk/hb_desk_ext/yolov11cpp/models/id_ki_v1.2.pt.torchscript",
      "/media/hbdesk/hb_desk_ext/yolov11cpp/models/id_ki_classes.txt", 0.7, 0.7,
      640, 640);
  cv::Mat image = cv::imread("/media/hbdesk/hb_desk_ext/kh_id_keyinfo/"
                             "combined_dset_kh2/asaasd123.jpg");
  if (image.empty()) {
    fprintf(stderr, "Failed to load image\n");
    return -1;
  };

  // torch::Tensor input_tensor;
  // preprocess(&image, &input_tensor, 640);
  // std::vector<torch::jit::IValue> _inputs = {};

  std::vector<Detection> _a = run_inference(keyinfo_model, &image);
}
