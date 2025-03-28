// header-only library for loading generic onnx models.
#ifndef ONNX_UTILS_H
#define ONNX_UTILS_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <onnxruntime_c_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

  /**
   * @brief onnx model abstraction.
   *
   * @param env `OrtEnv`.
   * @param session `OrtSession`.
   * @param session_options `OrtSessionOptions`
   * @param allocator `OrtAllocator`.
   */
  typedef struct
  {
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *session;
    OrtSessionOptions *session_options;
    OrtAllocator *allocator;
  } OnnxModel;

  // should we just put this in `OnnxModel` bro...
  typedef struct
  {
    size_t num_input_nodes;
    size_t num_output_nodes;
    char input_names;
    char output_names;
    OrtMemoryInfo *mem_info;
  } OnnxModelInfo;

  static inline void
  onnx_model_check_status (const OrtApi *api, OrtStatus *status)
  {
    if (status != NULL)
      {
        const char *msg = api->GetErrorMessage (status);
        fprintf (stderr, "Error: %s\n", msg);
        api->ReleaseStatus (status);
        exit (1);
      }
  }

  /// gets input/output names, dims and memory info
  static inline OnnxModelInfo
  onnx_model_get_info (OnnxModel *model)
  {
    if (!model)
      {
        fprintf (stderr, "INVALID OnnxModel!");
      }
    size_t num_input_nodes;
    size_t num_output_nodes;
    char *input_names;
    char *output_names;
    OrtMemoryInfo *mem_info;
    OrtStatus *status = NULL;
    status
        = model->api->SessionGetInputCount (model->session, &num_input_nodes);
    onnx_model_check_status (model->api, status);
    status = model->api->SessionGetInputName (model->session, 0,
                                              model->allocator, &input_names);
    // printf("[DEBUG] load shit : %s\n", input_names);
    onnx_model_check_status (model->api, status);

    status = model->api->SessionGetOutputCount (model->session,
                                                &num_output_nodes);
    onnx_model_check_status (model->api, status);

    status = model->api->SessionGetOutputName (
        model->session, 0, model->allocator, &output_names);

    // printf("[DEBUG] load output shit : %s\n", output_names);
    onnx_model_check_status (model->api, status);
    model->api->CreateCpuMemoryInfo (OrtArenaAllocator, OrtMemTypeDefault,
                                     &mem_info);
    onnx_model_check_status (model->api, status);
    return OnnxModelInfo{ .num_input_nodes = num_input_nodes,
                          .num_output_nodes = num_output_nodes,
                          .input_names = *input_names,
                          .output_names = *output_names,
                          .mem_info = mem_info };
  }

  static inline OnnxModel *
  onnx_model_load (const char *model_path)
  {
    OnnxModel *model = (OnnxModel *)malloc (sizeof (OnnxModel));
    if (!model)
      {
        fprintf (stderr, "failed to allocate memory for model!\n");
        return NULL;
      }
    memset (model, 0, sizeof (OnnxModel));

    model->api = OrtGetApiBase ()->GetApi (ORT_API_VERSION);
    if (!model->api)
      {
        fprintf (stderr, "failed to get onnxruntime api!\n");
        free (model);
        return NULL;
      }

    OrtStatus *status = model->api->CreateEnv (ORT_LOGGING_LEVEL_WARNING,
                                               "onnx-model", &model->env);
    if (status != NULL)
      {
        onnx_model_check_status (model->api, status);
        free (model);
        return NULL;
      }

    status = model->api->CreateSessionOptions (&model->session_options);
    onnx_model_check_status (model->api, status);

    status = model->api->SetSessionGraphOptimizationLevel (
        model->session_options, ORT_ENABLE_BASIC);
    onnx_model_check_status (model->api, status);

    status = model->api->CreateSession (
        model->env, model_path, model->session_options, &model->session);
    onnx_model_check_status (model->api, status);

    status = model->api->GetAllocatorWithDefaultOptions (&model->allocator);
    onnx_model_check_status (model->api, status);

    fprintf (stdout, "model successfully loaded!\n");
    return model;
  }

  static inline void
  onnx_model_free (OnnxModel *model)
  {
    if (model)
      {
        if (model->session)
          {
            model->api->ReleaseSession (model->session);
          }
        if (model->session_options)
          {
            model->api->ReleaseSessionOptions (model->session_options);
          }
        if (model->env)
          {
            model->api->ReleaseEnv (model->env);
          }
      }
    free (model);
  }

#ifdef __cplusplus
}
#endif

#endif /// onnx utils header
