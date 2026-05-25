#include "audio_utils.h"
#include <math.h>
#include <string.h>

void audio_normalize(float* samples, int len, float target_peak) {
    if (!samples || len <= 0) return;

    // Find peak
    float peak = 0.0f;
    for (int i = 0; i < len; i++) {
        float abs_val = fabsf(samples[i]);
        if (abs_val > peak) peak = abs_val;
    }

    if (peak < 1e-6f) return;  // silence

    float gain = target_peak / peak;

    for (int i = 0; i < len; i++) {
        samples[i] *= gain;
        // Clamp
        if (samples[i] > 1.0f) samples[i] = 1.0f;
        if (samples[i] < -1.0f) samples[i] = -1.0f;
    }
}

float audio_peak(const float* samples, int len) {
    if (!samples || len <= 0) return 0.0f;
    float peak = 0.0f;
    for (int i = 0; i < len; i++) {
        float abs_val = fabsf(samples[i]);
        if (abs_val > peak) peak = abs_val;
    }
    return peak;
}
