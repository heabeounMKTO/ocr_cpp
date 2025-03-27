# include_directories( 
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch/include
# )
cmake_minimum_required(VERSION 3.18)
project(yolo_detector LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_LINKER /usr/local/bin/mold)
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=mold")
set (CMAKE_CXX_FLAGS "-fuse-ld=mold")

message(STATUS "USING LINKER ${CMAKE_EXE_LINKER_FLAGS}")


set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(EXTERNAL_BUILD_DIR ${EXTERNAL_DIR}/build)
set(OPENCV_INSTALL_DIR ${EXTERNAL_DIR}/opencv)

set(PADDLE_LIB ${CMAKE_CURRENT_SOURCE_DIR}/external/paddle)
set(PADDLE_INCLUDE_DIR "${PADDLE_LIB}/paddle/include")
set(PADDLE_LIB_DIR "${PADDLE_LIB}/paddle/lib")
link_directories(PADDLE_LIB_DIR)

include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
include_directories("${PADDLE_LIB}/third_party/install/onnxruntime/include")
include_directories("${PADDLE_LIB}/third_party/install/paddle2onnx/include")
include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")

link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")
link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/third_party/install/onnxruntime/lib")
link_directories("${PADDLE_LIB}/third_party/install/paddle2onnx/lib")
link_directories("${PADDLE_LIB}/paddle/lib")

set(PADDLE ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})

set(ONNXRUNTIME_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime-linux-x64-1.21.0)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)
set(ONNXRUNTIME_LIB "libonnxruntime.so")
link_directories(${ONNXRUNTIME_LIB_DIR})
find_library(ONNXRUNTIME_LIB onnxruntime_providers_shared PATHS ${ONNXRUNTIME_LIB})


set(TORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch)
set(OPENCV_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/opencv-4.11.0)
set(OPENCV_CONTRIB ${CMAKE_CURRENT_SOURCE_DIR}/external/opencv_contrib-4/modules)



set(OPENCV_BUILD_DIR ${OPENCV_ROOT}/build)


if(NOT EXISTS ${OPENCV_INSTALL_DIR})
  file(MAKE_DIRECTORY ${OPENCV_INSTALL_DIR}) 
endif()

# message(OPENCV_BUILD_DIR="${OPENCV_BUILD_DIR}")

# build opencv from source
if(NOT EXISTS ${OPENCV_BUILD_DIR})
  message(STATUS "CONFIGURING OPENCV")
  execute_process(
    WORKING_DIRECTORY ${OPENCV_ROOT}
    COMMAND ${CMAKE_COMMAND}
    -B build 
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4/modules
    -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR}
    -DBUILD_SHARED_LIBS=OFF 
    -DWITH_CUDA=OFF 
    -DBUILD_opencv_python2=OFF
    -DBUILD_opencv_python3=OFF
    -DBUILD_JAVA=OFF
    -DBUILD_FAT_JAVA_LIB=OFF
    -DBUILD_opencv_java=OFF
    -DWITH_GTK=OFF
    -DENABLE_LTO=ON
    -DWITH_OPENGL=OFF 
    -DWITH_TBB=ON 
    -DWITH_DNN=OFF
    -DWITH_V4L=ON 
    -DWITH_FFMPEG=ON 
    -DBUILD_TESTS=OFF 
    -DBUILD_EXAMPLES=OFF 
    )
  execute_process(
    WORKING_DIRECTORY ${OPENCV_ROOT}
    COMMAND ${CMAKE_COMMAND} --build build --config Release --parallel 
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --install ${OPENCV_ROOT}/build --prefix ${OPENCV_INSTALL_DIR}
  )
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} {OPENCV_ROOT})
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${ONNXRUNTIME_ROOT})
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${TORCH_ROOT})

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(ONNXRuntime REQUIRED)
include_directories( 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TORCH_ROOT}/include
  ${TORCH_ROOT}/include/torch/csrc/api/include
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${OPENCV_ROOT/include/opencv4/opencv2}
  ${PADDLE_INCLUDE_DIR}
)

file (GLOB_RECURSE MODULE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# add_executable(yolo_torchscript src/main.cpp)
add_executable(run_ocr src/run_ocr.cpp)
target_link_libraries(run_ocr m ${ONNXRUNTIME_LIB} ${OpenCV_LIBS} ${PADDLE}) 

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

