#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ONNX encoder handle (opaque)
typedef struct OnnxEncoder OnnxEncoder;

// Encoder output: latent vectors for decoder input
struct EncoderOutput {
    float* z;          // latent vector [1, 192, T]
    int z_len;         // total float count in z
    float* y_mask;     // mask [1, 1, T]
    int mask_len;      // total float count in y_mask
    int T;             // sequence length dimension
};

// Initialize ONNX encoder from .onnx model file
// g_path: path to speaker embedding file (g.bin), can be NULL
OnnxEncoder* encoder_init(const char* model_path, const char* g_path);

// Run encoder inference on CPU
// phones/tones/langs: phoneme sequences from PinyinToPhoneme
// noise_scale: VITS noise scale (0.3 typical)
// length_scale: speed control (1.0 = normal)
// Output: allocated EncoderOutput (caller must free with encoder_free_output)
int encoder_run(OnnxEncoder* enc,
                const int* phones, const int* tones, const int* langs,
                int seq_len,
                float noise_scale, float length_scale,
                struct EncoderOutput* output);

// Free encoder output buffers
void encoder_free_output(struct EncoderOutput* output);

// Release ONNX encoder resources
void encoder_release(OnnxEncoder* enc);

#ifdef __cplusplus
}
#endif
