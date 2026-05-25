package com.nputts.rk3576

/**
 * NPU TTS JNI 接口
 *
 * C 层实现: ONNX Runtime encoder (CPU) + RKNN decoder (NPU)
 * 加载 libnpu_tts_jni.so
 */
object NpuTtsNative {
    init {
        System.loadLibrary("npu_tts_jni")
    }

    /**
     * 初始化模型
     * @param encoderPath encoder.onnx 路径 (CPU 推理)
     * @param decoderPath decoder_rk3576.rknn 路径 (NPU 推理)
     * @param gPath g.bin 路径 (speaker embedding, 可传空串)
     * @return native handle, 0 表示失败
     */
    external fun nativeInit(encoderPath: String, decoderPath: String,
                            gPath: String): Long

    /**
     * 合成音频
     * @param handle nativeInit 返回的 handle
     * @param phones 音素 ID 数组 (PinyinToPhoneme 输出)
     * @param tones 声调数组
     * @param langs 语言 ID 数组
     * @param noiseScale 噪声尺度 (0.3 = 自然)
     * @param lengthScale 语速控制 (1.0 = 正常)
     * @return PCM float32 数组, null 表示失败
     */
    external fun nativeSynthesize(handle: Long,
                                  phones: IntArray, tones: IntArray,
                                  langs: IntArray,
                                  noiseScale: Float, lengthScale: Float): FloatArray?

    /**
     * 释放模型资源
     */
    external fun nativeRelease(handle: Long)
}
