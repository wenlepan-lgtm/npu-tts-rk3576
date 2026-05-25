#include "rknn_decoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Conditional compilation: use real RKNN or stub
#ifdef USE_RKNN
#include <rknn_api.h>
#else
// Stub declarations from rknn_stub.c
typedef void* rknn_context;
extern "C" int rknn_init(rknn_context* ctx, const char* path, void* mem, size_t size, int flag);
extern "C" int rknn_destroy(rknn_context ctx);
extern "C" int rknn_run(rknn_context ctx, void* rknn_input, void* rknn_output);
#endif

struct RknnDecoder {
    rknn_context ctx;
    int initialized;
};

RknnDecoder* rknn_decoder_init(const char* model_path) {
    if (!model_path) return nullptr;

    RknnDecoder* dec = (RknnDecoder*)calloc(1, sizeof(RknnDecoder));
    if (!dec) return nullptr;

    int ret = rknn_init(&dec->ctx, model_path, nullptr, 0, 0);
    if (ret != 0) {
        fprintf(stderr, "rknn_decoder: rknn_init failed (ret=%d)\n", ret);
        free(dec);
        return nullptr;
    }

    dec->initialized = 1;
    return dec;
}

int rknn_decoder_run(RknnDecoder* dec,
                      const float* z, int z_len,
                      const float* y_mask, int mask_len,
                      float** out_samples, int* out_sample_count) {
    if (!dec || !dec->initialized || !z || !y_mask || !out_samples || !out_sample_count) {
        return -1;
    }

    // Build RKNN inputs
    rknn_input inputs[2];
    memset(inputs, 0, sizeof(inputs));

    // Input 0: z (latent vector)
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].size = z_len * sizeof(float);
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = (void*)z;

    // Input 1: y_mask
    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_FLOAT32;
    inputs[1].size = mask_len * sizeof(float);
    inputs[1].fmt = RKNN_TENSOR_NHWC;
    inputs[1].buf = (void*)y_mask;

    int ret = rknn_inputs_set(dec->ctx, 2, inputs);
    if (ret != 0) {
        fprintf(stderr, "rknn_decoder: rknn_inputs_set failed (ret=%d)\n", ret);
        return -1;
    }

    // Run inference
    ret = rknn_run(dec->ctx, nullptr);
    if (ret != 0) {
        fprintf(stderr, "rknn_decoder: rknn_run failed (ret=%d)\n", ret);
        return -1;
    }

    // Get outputs
    rknn_output output;
    memset(&output, 0, sizeof(output));
    output.index = 0;
    output.want_float = 1;

    ret = rknn_outputs_get(dec->ctx, 1, &output, nullptr);
    if (ret != 0) {
        fprintf(stderr, "rknn_decoder: rknn_outputs_get failed (ret=%d)\n", ret);
        return -1;
    }

    // Copy output data
    int sample_count = output.size / sizeof(float);
    float* samples = (float*)malloc(sample_count * sizeof(float));
    if (!samples) {
        rknn_outputs_release(dec->ctx, 1, &output);
        return -1;
    }
    memcpy(samples, output.buf, output.size);

    *out_samples = samples;
    *out_sample_count = sample_count;

    rknn_outputs_release(dec->ctx, 1, &output);
    return 0;
}

void rknn_decoder_release(RknnDecoder* dec) {
    if (!dec) return;
    if (dec->initialized) {
        rknn_destroy(dec->ctx);
    }
    free(dec);
}
