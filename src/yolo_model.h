#include <algorithm>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <torch/script.h>
#include <torch/torch.h>
#define INFERENCE_SIZE 640.0
#define BATCH_SIZE 1

using torch::indexing::Slice;
using torch::indexing::None;


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


static inline void scale_wh(int original_width, int original_height, 
                           int* new_width, int* new_height, float* ratio, 
                           const int target_size = 640) {
    float r = std::min(
        static_cast<float>(target_size) / original_width,
        static_cast<float>(target_size) / original_height
    );
    *new_width = std::round(original_width * r);
    *new_height = std::round(original_height * r);
    *ratio = r;
}

torch::Tensor xyxy2xywh(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    y.index_put_({"...", 0}, (x.index({"...", 0}) + x.index({"...", 2})).div(2));
    y.index_put_({"...", 1}, (x.index({"...", 1}) + x.index({"...", 3})).div(2));
    y.index_put_({"...", 2}, x.index({"...", 2}) - x.index({"...", 0}));
    y.index_put_({"...", 3}, x.index({"...", 3}) - x.index({"...", 1}));
    return y;
}

torch::Tensor xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    auto dw = x.index({"...", 2}).div(2);
    auto dh = x.index({"...", 3}).div(2);
    y.index_put_({"...", 0}, x.index({"...", 0}) - dw);
    y.index_put_({"...", 1}, x.index({"...", 1}) - dh);
    y.index_put_({"...", 2}, x.index({"...", 0}) + dw);
    y.index_put_({"...", 3}, x.index({"...", 1}) + dh);
    return y;
}

/* preprocess into letterbox format */
static inline float preprocess(cv::Mat *input_image, 
                              torch::Tensor *output_tensor, 
                              const int target_size = 640) {
    // Get original dimensions
    int o_width = input_image->cols;
    int o_height = input_image->rows;

    int new_w, new_h;
    float ratio;
    scale_wh(o_width, o_height, &new_w, &new_h, &ratio, target_size);

    cv::Mat canvas = cv::Mat(target_size, target_size, input_image->type(), cv::Scalar(114, 114, 114));
    
    cv::Mat resized;
    cv::resize(*input_image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);
    
    int dx = (target_size - new_w) / 2;
    int dy = (target_size - new_h) / 2;

    cv::Mat roi = canvas(cv::Rect(dx, dy, new_w, new_h));
    resized.copyTo(roi);
    
    // cv::imwrite("./letterbox_debug.png", canvas);

    cv::Mat float_img;
    canvas.convertTo(float_img, CV_32F, 1.0/255.0);
    *output_tensor = torch::from_blob(float_img.data, {target_size, target_size, 3}, torch::kFloat32).clone();
    *output_tensor = output_tensor->permute({2, 0, 1}).contiguous();
    *output_tensor = output_tensor->unsqueeze(0);
    return ratio;
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

static inline YoloModel *create_yolo_detector(const char *model_path,
                                              const char *classes_path, 
                                              const torch::Device device,
                                              float conf_thresh,
                                              float nms_thresh, 
                                              int w, int h) {
  YoloModel *detector = new YoloModel();
  try {
    detector->model = torch::jit::load(model_path);
    detector->model.eval();
    detector->model.to(device, torch::kFloat32);
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

  // torch::Device device(torch::kCPU);
  std::vector<Detection> detections;
  torch::Tensor input_tensor;

  float preproc_ratio = preprocess(input_image, &input_tensor, INFERENCE_SIZE);
  printf("input_image tensor shape: [");
  for (int i = 0; i < input_tensor.dim(); i++) {
    printf("%ld%s", input_tensor.size(i),
           i < input_tensor.dim() - 1 ? ", " : "");
  }
  printf("]\n");
  float conf_thresh = 0.5;
  float iou_thresh = 0.5;
  int max_det = 100;
  try {
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> _inputs;
    _inputs.push_back(input_tensor);
    auto output = detector->model.forward(_inputs).toTensor();
    output = output.to(torch::kCPU);

    /*  [1, 14, 8400]
     https://github.com/orgs/ultralytics/discussions/17254
    */
    int channels = output.size(1);
    int num_points = output.size(2);
    
    std::vector<Detection> detections;
    for (int i=0; i < num_points; i++) {
      auto objectness = output[0][4][i].item<float>();
      if (objectness >= detector->conf_thresh) {
        fprintf(stdout, "OBJECTNESS %f\n", objectness);
        float x = output[0][0][i].item<float>();
        float y = output[0][1][i].item<float>();
        float w = output[0][2][i].item<float>();
        float h = output[0][3][i].item<float>();
        std::vector<float> class_scores;

        // starts at 5 due to channels - 5 = class conf
        for (int cls = 5; cls < channels; cls++) {
          class_scores.push_back(output[0][cls][i].item<float>());
        }

        auto max_it = std::max_element(class_scores.begin(), class_scores.end());
        int class_id = std::distance(class_scores.begin(), max_it);
        float class_conf = *max_it;
        float final_conf = objectness * class_conf;
        fprintf(stdout, "final conf: %f\n", final_conf);

      } 
    }
    
    return detections;

  } catch (const std::exception &e) {
    fprintf(stderr, "fucking torch forward error %s", e.what());
    throw;
  }
}

