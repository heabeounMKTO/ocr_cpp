#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <paddle_inference_api.h>
#include <vector>

typedef struct
{
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
} PaddleCharacterRecognizerConfig;

static inline void
permute (const cv::Mat &im, float *data) noexcept
{
  int rh = im.rows;
  int rw = im.cols;
  int rc = im.channels ();

  for (int i = 0; i < rc; i++)
    {
      cv::extractChannel (im, cv::Mat (rh, rw, CV_32FC1, data + i * rh * rw),
                          i);
    }
}

static inline void
rec_resize_input (cv::Mat &img, cv::Mat &resize_img, float wh_ratio,
                  bool use_tensorrt,
                  const std::vector<int> &rec_image_shape) noexcept
{
  int imgC, imgH, imgW;
  imgC = rec_image_shape[0];
  imgH = rec_image_shape[1];
  imgW = rec_image_shape[2];
  imgW = int (imgH * wh_ratio);
  float ratio = float (img.cols) / float (img.rows);
  int resize_w, resize_h;
  if (ceilf (imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int (ceilf (imgH * ratio));

  cv::resize (img, resize_img, cv::Size (resize_w, imgH), 0.f, 0.f,
              cv::INTER_LINEAR);
  cv::copyMakeBorder (resize_img, resize_img, 0, 0, 0,
                      int (imgW - resize_img.cols), cv::BORDER_CONSTANT,
                      { 0, 0, 0 });
}

static inline void
rec_normalize_input (cv::Mat &im, const std::vector<float> &mean,
                     const std::vector<float> &scale, const bool is_scale)
{
  double e = 1.0;
  if (is_scale)
    {
      e /= 255.0;
    }
  im.convertTo (im, CV_32FC3, e);
  std::vector<cv::Mat> bgr_channels (3);
  cv::split (im, bgr_channels);
  for (size_t i = 0; i < bgr_channels.size (); ++i)
    {
      bgr_channels[i].convertTo (bgr_channels[i], CV_32FC1, 1.0 * scale[i],
                                 (0.0 - mean[i]) * scale[i]);
    }
  cv::merge (bgr_channels, im);
}

static inline std::vector<std::string>
read_dict (const std::string &path) noexcept
{
  std::vector<std::string> m_vec;
  std::ifstream in (path);
  if (in)
    {
      for (;;)
        {
          std::string line;
          if (!getline (in, line))
            {
              break;
            }
          m_vec.emplace_back (std::move (line));
        }
    }
  else
    {
      std::cout << "label file not found! : " << path << "\n" << std::endl;
      exit (1);
    }
  return m_vec;
}

static inline std::shared_ptr<paddle_infer::Predictor>
new_paddle_recognizer (const std::string model_dir, const bool use_gpu,
                       const int gpu_id, const int gpu_mem,
                       const int cpu_math_library_num_threads,
                       const bool use_mkldnn, const std::string label_path,
                       const bool use_tensorrt, const std::string precision,
                       const int rec_batch_num, const int rec_img_h,
                       const int rec_img_w)
{
  paddle_infer::Config config;
  config.SetModel (model_dir + "/inference.pdmodel",
                   model_dir + "/inference.pdiparams");
  std::cout << "In PP-OCRv3, default rec_img_h is 48,"
            << "if you use other model, you should set the param rec_img_h=32"
            << std::endl;
  // add gpu suppourt later , its optional
  config.DisableGpu ();
  config.SetCpuMathLibraryNumThreads (cpu_math_library_num_threads);
  auto pass_builder = config.pass_builder ();
  pass_builder->DeletePass ("matmul_transpose_reshape_fuse_pass");
  pass_builder->DeletePass ("matmul_transpose_reshape_fuse_pass");
  config.SwitchUseFeedFetchOps (false);
  // true for multiple input
  config.SwitchSpecifyInputNames (true);

  config.SwitchIrOptim (true);

  config.EnableMemoryOptim ();
  //   config.DisableGlogInfo();
  std::shared_ptr<paddle_infer::Predictor> pred
      = paddle_infer::CreatePredictor (config);
  return pred;
}

static inline void
run_rec_model (std::shared_ptr<paddle_infer::Predictor> &char_rec,
               const std::vector<cv::Mat> &input_img)
{
}
