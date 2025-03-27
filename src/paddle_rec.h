#include <paddle_inference_api.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

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

static inline std::vector<std::string>
read_dict(const std::string &path) noexcept {
  std::vector<std::string> m_vec;
  std::ifstream in(path);
  if (in) {
    for (;;) {
      std::string line;
      if (!getline(in, line)) {
        break;
      }
      m_vec.emplace_back(std::move(line));
    }
  } else {
    std::cout << "label file not found! : " << path << "\n" << std::endl;
    exit(1);
  }
  return m_vec;
}

static inline std::shared_ptr<paddle_infer::Predictor> new_paddle_recognizer(
    const std::string model_dir, const bool use_gpu, const int gpu_id,
    const int gpu_mem, const int cpu_math_library_num_threads,
    const bool use_mkldnn, const std::string label_path,
    const bool use_tensorrt, const std::string precision,
    const int rec_batch_num, const int rec_img_h, const int rec_img_w) {
  paddle_infer::Config config;
  config.SetModel(model_dir + "/inference.pdmodel",
                  model_dir + "/inference.pdiparams");
  std::cout << "In PP-OCRv3, default rec_img_h is 48,"
            << "if you use other model, you should set the param rec_img_h=32"
            << std::endl;
  // add gpu suppourt later , its optional
  config.DisableGpu();
  config.SetCpuMathLibraryNumThreads(cpu_math_library_num_threads);
  auto pass_builder = config.pass_builder();
  pass_builder->DeletePass("matmul_transpose_reshape_fuse_pass");
  pass_builder->DeletePass("matmul_transpose_reshape_fuse_pass");
  config.SwitchUseFeedFetchOps(false);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  //   config.DisableGlogInfo();
  std::shared_ptr<paddle_infer::Predictor> pred =
      paddle_infer::CreatePredictor(config);
  return pred;
}
