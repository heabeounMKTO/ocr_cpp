#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

typedef struct {
    float x, y, width, height;
} BBox;

typedef struct {
  int class_id;
  float confidence;
  cv::Rect box;
} Detection;

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


static inline std::vector<std::string> get_class_names(const std::string &path) {
        std::vector<std::string> classNames;
        std::ifstream infile(path);
        if (infile) {
            std::string line;
            while (getline(infile, line)) {
                // Remove carriage return if present (for Windows compatibility)
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                classNames.emplace_back(line);
            }
        } else {
            std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
        }

        std::cout << "Loaded " << classNames.size() << " class names from " + path << std::endl;
        return classNames;
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

static inline cv::Mat letterbox(cv::Mat *input_image, int target_size = 640) {
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

    cv::Mat float_img;
    canvas.convertTo(float_img, CV_32F, 1.0/255.0);
    return canvas;
}


