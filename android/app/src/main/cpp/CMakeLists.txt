cmake_minimum_required(VERSION 3.22)

project("npu-tts-rk3576")

set(CMAKE_CXX_STANDARD 17)

# --- ONNX Runtime ---
# Expected at app/src/main/jniLibs/${ANDROID_ABI}/libonnxruntime.so
# Header: ONNX Runtime SDK should be in cpp/onnxruntime/
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")
if(EXISTS "${ONNXRUNTIME_DIR}/include")
    include_directories("${ONNXRUNTIME_DIR}/include")
else()
    message(WARNING "ONNX Runtime headers not found at ${ONNXRUNTIME_DIR}/include")
endif()

# --- RKNN Runtime ---
# librknnrt.so is in jniLibs, header in cpp/rknn/
set(RKNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rknn")
if(EXISTS "${RKNN_DIR}/rknn_api.h")
    include_directories("${RKNN_DIR}")
    add_definitions(-DUSE_RKNN)
else()
    message(WARNING "RKNN headers not found, using stubs (NPU will not work)")
endif()

# --- Source files ---
add_library(npu_tts_jni SHARED
    npu_tts_jni.cpp
    rknn_decoder.cpp
    onnx_encoder.cpp
    audio_utils.cpp
)

# Stub for non-RKNN builds
if(NOT EXISTS "${RKNN_DIR}/rknn_api.h")
    target_sources(npu_tts_jni PRIVATE rknn_stub.c)
endif()

# --- Link libraries ---
find_library(log-lib log)

# ONNX Runtime: link against the .so in jniLibs
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libonnxruntime.so")

# RKNN Runtime: link against the .so in jniLibs (if available)
if(EXISTS "${RKNN_DIR}/rknn_api.h")
    add_library(rknnrt SHARED IMPORTED)
    set_target_properties(rknnrt PROPERTIES
        IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/librknnrt.so")
    target_link_libraries(npu_tts_jni rknnrt)
endif()

target_link_libraries(npu_tts_jni
    onnxruntime
    ${log-lib}
)

# JNI header path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})
