#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// RKNN decoder handle (opaque)
typedef struct RknnDecoder RknnDecoder;

// Initialize RKNN decoder from .rknn model file
// Returns NULL on failure
RknnDecoder* rknn_decoder_init(const char* model_path);

// Run decoder inference on NPU
// Inputs: z (latent vector), y_mask (mask)
// Output: audio samples written to out_samples, returns sample count
// Returns -1 on failure
int rknn_decoder_run(RknnDecoder* dec,
                      const float* z, int z_len,
                      const float* y_mask, int mask_len,
                      float** out_samples, int* out_sample_count);

// Release RKNN decoder resources
void rknn_decoder_release(RknnDecoder* dec);

#ifdef __cplusplus
}
#endif
