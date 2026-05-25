package com.nputts.rk3576

import android.util.Log
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * NPU TTS 引擎
 *
 * Encoder: ONNX Runtime CPU
 * Decoder: RKNN NPU (RK3576)
 *
 * 用法:
 *   val engine = NpuTtsEngine()
 *   engine.init("/sdcard/npu-tts-models")
 *   engine.synthesize("你好世界") { pcm ->
 *       // pcm: FloatArray, 44100Hz mono
 *   }
 *   engine.release()
 */
class NpuTtsEngine {
    companion object {
        private const val TAG = "NpuTtsEngine"
        const val SAMPLE_RATE = 44100
    }

    private var nativeHandle: Long = 0
    private val ready = AtomicBoolean(false)
    private val speaking = AtomicBoolean(false)
    private var speed = 0.9f

    /**
     * 初始化引擎，加载模型
     * @param modelDir 包含 encoder.onnx, decoder_rk3576.rknn, g.bin 的目录
     * @return true 成功
     */
    fun init(modelDir: String): Boolean {
        val encoderFile = File(modelDir, "encoder.onnx")
        val decoderFile = File(modelDir, "decoder_rk3576.rknn")
        val gFile = File(modelDir, "g.bin")

        if (!encoderFile.exists()) {
            Log.e(TAG, "encoder.onnx not found: ${encoderFile.absolutePath}")
            return false
        }
        if (!decoderFile.exists()) {
            Log.e(TAG, "decoder_rk3576.rknn not found: ${decoderFile.absolutePath}")
            return false
        }

        val gPath = if (gFile.exists()) gFile.absolutePath else ""

        Log.i(TAG, "Loading models from $modelDir")
        val t0 = System.currentTimeMillis()

        nativeHandle = NpuTtsNative.nativeInit(
            encoderFile.absolutePath,
            decoderFile.absolutePath,
            gPath
        )

        val elapsed = System.currentTimeMillis() - t0
        if (nativeHandle == 0L) {
            Log.e(TAG, "nativeInit failed")
            return false
        }

        ready.set(true)
        Log.i(TAG, "NPU TTS ready (${elapsed}ms)")
        return true
    }

    /**
     * 合成音频
     * @param text 中文文本
     * @return PCM float32 @ SAMPLE_RATE Hz, null 表示失败
     */
    fun synthesize(text: String): FloatArray? {
        if (!ready.get() || nativeHandle == 0L) {
            Log.w(TAG, "Engine not ready")
            return null
        }

        speaking.set(true)
        try {
            // 1. Text → phonemes
            val phonemes = PinyinToPhoneme.convert(text)
            Log.i(TAG, "Synthesizing: '${text.take(30)}' (${phonemes.phones.size} tokens)")

            // 2. Encoder (CPU) + Decoder (NPU)
            val noiseScale = 0.3f
            val lengthScale = 1.0f / speed

            val t0 = System.currentTimeMillis()
            val pcm = NpuTtsNative.nativeSynthesize(
                nativeHandle,
                phonemes.phones, phonemes.tones, phonemes.langs,
                noiseScale, lengthScale
            )
            val elapsed = System.currentTimeMillis() - t0

            if (pcm == null) {
                Log.e(TAG, "Synthesis failed")
                return null
            }

            val duration = pcm.size / SAMPLE_RATE.toFloat()
            val rtf = elapsed / 1000f / duration
            Log.i(TAG, "Done: ${elapsed}ms, ${pcm.size} samples (${duration.format(2)}s, RTF=${rtf.format(3)})")

            return pcm
        } catch (e: Exception) {
            Log.e(TAG, "Synthesis error", e)
            return null
        } finally {
            speaking.set(false)
        }
    }

    fun setSpeed(s: Float) {
        speed = s.coerceIn(0.5f, 2.0f)
    }

    fun isReady(): Boolean = ready.get()
    fun isSpeaking(): Boolean = speaking.get()

    fun release() {
        speaking.set(false)
        ready.set(false)
        if (nativeHandle != 0L) {
            NpuTtsNative.nativeRelease(nativeHandle)
            nativeHandle = 0
        }
        Log.i(TAG, "NPU TTS released")
    }

    private fun Float.format(digits: Int) = String.format("%.${digits}f", this)
}
