#include "yolo_model.h"
#include <cstdio>


int main(int argc, char* argv[]) {
  struct YoloContext *ctx; 
  int init_ok  = load_model(ctx, "/media/hbdesk/hb_desk_ext/ocr_core/models/id_ki_v1.1.onnx");
  printf("fuqq %d", 1);
  
}
