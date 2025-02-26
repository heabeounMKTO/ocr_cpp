# include_directories( 
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch/include
# )
cmake_minimum_required(VERSION 3.18)
project(yolo_detector LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LINKER /usr/local/bin/mold)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=mold")

set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(EXTERNAL_BUILD_DIR ${EXTERNAL_DIR}/build)
set(OPENCV_INSTALL_DIR ${EXTERNAL_DIR}/opencv)

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
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${TORCH_ROOT})

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories( 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TORCH_ROOT}/include
  ${TORCH_ROOT}/include/torch/csrc/api/include
  ${OPENCV_ROOT/include/opencv4/opencv2}
)

file (GLOB_RECURSE MODULE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_executable(yolo_torchscript src/main.cpp)
target_link_libraries(yolo_torchscript ${TORCH_LIBRARIES} ${OpenCV_LIBS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

