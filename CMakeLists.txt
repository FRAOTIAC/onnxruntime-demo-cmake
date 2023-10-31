cmake_minimum_required(VERSION 3.12)
set (CMAKE_CXX_STANDARD 17)


project(onnxruntime_demo_gpu C CXX)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")

#onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_NNAPI_BUILTIN "Build with builtin NNAPI lib for Android NNAPI support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_NUPHAR "Build with Nuphar" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(LIBPNG_ROOTDIR "libpng root dir")
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

include_directories("/usr/local/include/onnxruntime")
set(ONNXRUNTIME_ROOTDIR "/usr/local")

include_directories(
        "${ONNXRUNTIME_ROOTDIR}/include"
        "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"
        "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/providers/cpu"
)
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")


if (onnxruntime_USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif ()
if (onnxruntime_USE_OPENVINO)
    add_definitions(-DUSE_OPENVINO)
endif ()
if (onnxruntime_USE_NNAPI_BUILTIN)
    add_definitions(-DUSE_NNAPI)
endif ()
if (onnxruntime_USE_DNNL)
    add_definitions(-DUSE_DNNL)
endif ()
if (onnxruntime_USE_NUPHAR)
    add_definitions(-DUSE_NUPHAR)
endif ()
if (onnxruntime_USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
endif ()
if (onnxruntime_USE_DML)
    message("Enabling DML")
    add_definitions(-DUSE_DML)
endif ()

add_subdirectory(OnnxRuntimeDemoGPU)