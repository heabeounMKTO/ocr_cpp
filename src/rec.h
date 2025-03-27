#include <paddle_inference_api.h>
#include <iostream>
#include <chrono>
#include <iostream>
#include <numeric>

typedef struct {
  const std::string model_dir;
  const bool use_gpu;
  const int gpu_id; 
  const int gpu_mem; 
  const int cpu_math_library_num_threads;
  const bool use_mkldnn;
  const std::string label_path;
  const bool use_tensorrt;
  const std::string precision;
  const int rec_batch_num;
  const int rec_img_h;
  const int rec_img_w;
} PaddleCharacterRecognizer;

static inline void new_paddle_recognizer(
  PaddleCharacterRecognizer* pcr,
  const std::string model_dir,
  const bool use_gpu,
  const int gpu_id, 
  const int gpu_mem, 
  const int cpu_math_library_num_threads,
  const bool use_mkldnn,
  const std::string label_path,
  const bool use_tensorrt,
  const std::string precision,
  const int rec_batch_num,
  const int rec_img_h,
  const int rec_img_w) {
  paddle_infer::Config config; 
  config.SetModel(model_dir + "/inference.pdmodel", model_dir + "/inference.pdiparams");
  std::cout << "In PP-OCRv3, default rec_img_h is 48,"
            << "if you use other model, you should set the param rec_img_h=32"
            << std::endl;
} 
