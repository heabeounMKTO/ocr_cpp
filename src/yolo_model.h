#include <cstdio>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <torch/script.h>
#define INFERENCE_SIZE 640.0

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


static inline YoloModel *create_yolo_detector(const char *model_path,
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

static inline void scale_wh(int original_width, int original_height, 
                            int *target_width, int *target_height) {
  float r = std::min((float) INFERENCE_SIZE / original_width, 
                     (float) INFERENCE_SIZE / original_height);

  fprintf(stdout, "[debug] scale_wh R %f\n", r);
  fprintf(stdout, "[debug] mul w %f\n", original_width * r);
  fprintf(stdout, "[debug] mul h %f\n", original_height * r);
  *target_width = (int) std::round(original_width * r);

  *target_height = (int) std::round(original_height * r);
}

static inline void preprocess(cv::Mat *image, torch::Tensor *output, int target_size) {
  cv::Mat resized;
  cv::resize(*image, resized, cv::Size(target_size, target_size));
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  cv::Mat float_img;
  resized.convertTo(float_img, CV_32F, 1.0 / 255.0);
  *output = torch::from_blob(float_img.data, {1,target_size, target_size, 3}, torch::kFloat32).clone(); // dealloc aka not clonign causes a segfault
  *output = output->permute({0, 3, 1, 2}).contiguous();
}



static inline float compute_iou(cv::Rect *box1, cv::Rect *box2) {
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
