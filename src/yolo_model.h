#ifndef YOLO_MODEL_H
#define YOLO_MODEL_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <onnxruntime_cxx_api.h>
#include <cstdio>


#define SUCCESS 0
#define FAILURE 1

typedef struct {
  float x, y, w, h;
} BBox;


typedef struct {
    BBox box;
    float confidence;
    int class_id;
    const char* class_name;
} Detection;


 struct YoloContext{
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
  Ort::SessionOptions session_options{};
  std::unique_ptr<Ort::Session> session{nullptr};
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
  // Model information
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
  std::vector<std::string> input_names_str;
  std::vector<std::string> output_names_str;
  std::vector<Ort::Value> input_tensors;
  std::vector<int64_t> input_shape;
  size_t input_height;
  size_t input_width;
  size_t num_channels;
  
  // Class names
  std::vector<const char*> class_names;
  std::vector<std::string> class_names_str;
  int num_classes;
  
  // Buffer for preprocessed image
  std::vector<float> input_tensor_values;
}  ;

static inline int load_model(YoloContext* ctx, const char* model_path) {
    try {
        // Set session options
        ctx->session_options.SetIntraOpNumThreads(1);
        // ctx->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        return SUCCESS; 
    //     // Create session
    //     ctx->session = std::make_unique<Ort::Session>(ctx->env, model_path, ctx->session_options);
    //     
    //     // Get input information
    //     Ort::AllocatorWithDefaultOptions allocator;
    //     
    //     // Get number of inputs and outputs
    //     size_t num_input_nodes = ctx->session->GetInputCount();
    //     size_t num_output_nodes = ctx->session->GetOutputCount();
    //     
    //     // Process inputs
    //     ctx->input_names_str.reserve(num_input_nodes);
    //     ctx->input_names.reserve(num_input_nodes);
    //     
    //     // Get first input name
    //     auto input_name = ctx->session->GetInputNameAllocated(0, allocator);
    //     ctx->input_names_str.push_back(input_name.get());
    //     ctx->input_names.push_back(ctx->input_names_str.back().c_str());
    //     
    //     // Get input shape
    //     auto input_shape_info = ctx->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    //     ctx->input_shape = input_shape_info.GetShape();
    //     
    //     // Fix any dynamic dimensions (usually batch size)
    //     if (ctx->input_shape[0] == -1) {
    //         ctx->input_shape[0] = 1;
    //     }
    //     
    //     // Store dimensions for easier access
    //     ctx->num_channels = ctx->input_shape[1];
    //     ctx->input_height = ctx->input_shape[2];
    //     ctx->input_width = ctx->input_shape[3];
    //     
    //     // Process outputs
    //     ctx->output_names_str.reserve(num_output_nodes);
    //     ctx->output_names.reserve(num_output_nodes);
    //     
    //     // Get output name
    //     auto output_name = ctx->session->GetOutputNameAllocated(0, allocator);
    //     ctx->output_names_str.push_back(output_name.get());
    //     ctx->output_names.push_back(ctx->output_names_str.back().c_str());
    //     
    //     // Initialize class names (example - replace with actual class names)
    //     ctx->num_classes = 80; // COCO dataset has 80 classes
    //     
    //     // Sample class names for COCO - replace with actual classes for your model
    //     const char* coco_classes[] = {
    //         "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    //         "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    //         "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    //         "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    //         "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    //         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    //         "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    //         "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    //         "hair drier", "toothbrush"
    //     };
    //     
    //     ctx->class_names_str.reserve(ctx->num_classes);
    //     ctx->class_names.reserve(ctx->num_classes);
    //     
    //     for (int i = 0; i < ctx->num_classes; i++) {
    //         ctx->class_names_str.push_back(coco_classes[i]);
    //         ctx->class_names.push_back(ctx->class_names_str.back().c_str());
    //     }
    //     
    //     // Allocate memory for input tensor
    //     size_t input_tensor_size = 1;
    //     for (size_t i = 0; i < ctx->input_shape.size(); i++) {
    //         input_tensor_size *= ctx->input_shape[i];
    //     }
    //     ctx->input_tensor_values.resize(input_tensor_size);
    //     
    //     return SUCCESS;
    }
    catch (const Ort::Exception& e) {
        fprintf(stderr, "ONNX Runtime error: %s\n", e.what());
        return FAILURE;
    }
    catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return FAILURE;
    }
}

#endif
