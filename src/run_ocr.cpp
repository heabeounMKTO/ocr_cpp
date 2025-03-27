#include <iostream>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "paddle_inference_api.h"
#include "paddle_rec.h"
#include "yolo_model.h"

int main(int argc, char *argv[]) {
  const std::string model_path = "./models/id_ki_v12.onnx";
  const std::string image_path = "/home/hbdesk/Pictures/vn_passport/test1.jpg";
  const std::string labels_path = "./models/id_ki_classes.txt";
  const std::string save_path = "./test_image.jpeg";
  YOLO12Detector detector(model_path, labels_path, false);

  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    std::cerr << "couldnt open the image!\n";
    return -1;
  }
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<Detection> results = detector.detect(image);

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start);

  std::cout << "Detection completed in: " << duration.count() << " ms"
            << std::endl;

  std::shared_ptr<paddle_infer::Predictor> _a = new_paddle_recognizer(
      "/media/hbdesk/hb_desk_ext/ocr_core/models/"
      "latin_ppocr_mobile_v2.0_rec_infer",
      false, 0, 1024, 4, false,
      "/media/hbdesk/hb_desk_ext/ocr_core/models/kh_dict.txt", false, "f32", 1,
      112, 112);
  detector.drawBoundingBox(image, results); // Simple bounding box drawing
  // detector.drawBoundingBoxMask(image, results); // Uncomment for mask drawing

  // Save the processed image to the specified directory
  if (cv::imwrite(save_path, image)) {
    std::cout << "Processed image saved successfully at: " << save_path
              << std::endl;
  } else {
    std::cerr << "Error: Could not save the processed image to: " << save_path
              << std::endl;
  }
  return 0;
}
