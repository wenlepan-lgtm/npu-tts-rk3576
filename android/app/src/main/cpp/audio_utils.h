#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Normalize audio to target peak amplitude
// RKNN decoder output has very low amplitude (~-30dB), this compensates
void audio_normalize(float* samples, int len, float target_peak);

// Find peak absolute value in sample buffer
float audio_peak(const float* samples, int len);

#ifdef __cplusplus
}
#endif
