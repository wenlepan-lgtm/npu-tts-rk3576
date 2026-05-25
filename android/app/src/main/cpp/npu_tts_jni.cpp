#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rknn_decoder.h"
#include "onnx_encoder.h"
#include "audio_utils.h"

// Engine state: holds encoder + decoder
struct NpuTtsEngine {
    OnnxEncoder* encoder;
    RknnDecoder* decoder;
};

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_nputts_rk3576_NpuTtsNative_nativeInit(
        JNIEnv* env, jobject thiz,
        jstring encoder_path, jstring decoder_path, jstring g_path) {

    const char* enc_path = env->GetStringUTFChars(encoder_path, nullptr);
    const char* dec_path = env->GetStringUTFChars(decoder_path, nullptr);
    const char* gpath = g_path ? env->GetStringUTFChars(g_path, nullptr) : nullptr;

    auto* engine = new NpuTtsEngine();
    engine->encoder = nullptr;
    engine->decoder = nullptr;

    // Init encoder (CPU, ONNX Runtime)
    engine->encoder = encoder_init(enc_path, gpath);
    if (!engine->encoder) {
        fprintf(stderr, "NPU TTS: encoder init failed\n");
        delete engine;
        env->ReleaseStringUTFChars(encoder_path, enc_path);
        env->ReleaseStringUTFChars(decoder_path, dec_path);
        if (gpath) env->ReleaseStringUTFChars(g_path, gpath);
        return 0;
    }

    // Init decoder (NPU, RKNN Runtime)
    engine->decoder = rknn_decoder_init(dec_path);
    if (!engine->decoder) {
        fprintf(stderr, "NPU TTS: decoder init failed (RKNN)\n");
        encoder_release(engine->encoder);
        delete engine;
        env->ReleaseStringUTFChars(encoder_path, enc_path);
        env->ReleaseStringUTFChars(decoder_path, dec_path);
        if (gpath) env->ReleaseStringUTFChars(g_path, gpath);
        return 0;
    }

    env->ReleaseStringUTFChars(encoder_path, enc_path);
    env->ReleaseStringUTFChars(decoder_path, dec_path);
    if (gpath) env->ReleaseStringUTFChars(g_path, gpath);

    return reinterpret_cast<jlong>(engine);
}

JNIEXPORT jfloatArray JNICALL
Java_com_nputts_rk3576_NpuTtsNative_nativeSynthesize(
        JNIEnv* env, jobject thiz,
        jlong handle,
        jintArray phones, jintArray tones, jintArray langs,
        jfloat noise_scale, jfloat length_scale) {

    auto* engine = reinterpret_cast<NpuTtsEngine*>(handle);
    if (!engine || !engine->encoder || !engine->decoder) {
        return nullptr;
    }

    int seq_len = env->GetArrayLength(phones);
    jint* phones_arr = env->GetIntArrayElements(phones, nullptr);
    jint* tones_arr = env->GetIntArrayElements(tones, nullptr);
    jint* langs_arr = env->GetIntArrayElements(langs, nullptr);

    // Step 1: Run encoder on CPU
    EncoderOutput enc_out = {};
    int ret = encoder_run(engine->encoder,
                          reinterpret_cast<const int*>(phones_arr),
                          reinterpret_cast<const int*>(tones_arr),
                          reinterpret_cast<const int*>(langs_arr),
                          seq_len,
                          noise_scale, length_scale,
                          &enc_out);

    env->ReleaseIntArrayElements(phones, phones_arr, JNI_ABORT);
    env->ReleaseIntArrayElements(tones, tones_arr, JNI_ABORT);
    env->ReleaseIntArrayElements(langs, langs_arr, JNI_ABORT);

    if (ret != 0 || !enc_out.z || !enc_out.y_mask) {
        return nullptr;
    }

    // Step 2: Run decoder on NPU
    float* audio_samples = nullptr;
    int sample_count = 0;
    ret = rknn_decoder_run(engine->decoder,
                           enc_out.z, enc_out.z_len,
                           enc_out.y_mask, enc_out.mask_len,
                           &audio_samples, &sample_count);

    encoder_free_output(&enc_out);

    if (ret != 0 || !audio_samples || sample_count <= 0) {
        return nullptr;
    }

    // Step 3: Normalize audio (RKNN decoder output has low amplitude)
    audio_normalize(audio_samples, sample_count, 0.9f);

    // Step 4: Return to Java
    jfloatArray result = env->NewFloatArray(sample_count);
    if (result) {
        env->SetFloatArrayRegion(result, 0, sample_count, audio_samples);
    }
    free(audio_samples);

    return result;
}

JNIEXPORT void JNICALL
Java_com_nputts_rk3576_NpuTtsNative_nativeRelease(
        JNIEnv* env, jobject thiz, jlong handle) {

    auto* engine = reinterpret_cast<NpuTtsEngine*>(handle);
    if (!engine) return;

    if (engine->decoder) rknn_decoder_release(engine->decoder);
    if (engine->encoder) encoder_release(engine->encoder);
    delete engine;
}

}  // extern "C"
