package com.nputts.rk3576

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log

/**
 * AudioTrack 播放封装
 * 播放 TTS 引擎输出的 PCM float32 音频
 */
class AudioPlayer(
    private val sampleRate: Int = 44100
) {
    companion object {
        private const val TAG = "AudioPlayer"
    }

    private var audioTrack: AudioTrack? = null

    fun init() {
        val minBuf = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT
        )
        val bufSize = minBuf * 8

        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .setSampleRate(sampleRate)
                    .build()
            )
            .setBufferSizeInBytes(bufSize)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
    }

    fun play(pcm: FloatArray) {
        val track = audioTrack ?: return
        track.play()

        val chunkSize = 4096
        var offset = 0
        while (offset < pcm.size) {
            val len = (offset + chunkSize).coerceAtMost(pcm.size) - offset
            track.write(pcm, offset, len, AudioTrack.WRITE_BLOCKING)
            offset += len
        }

        track.stop()
    }

    fun stop() {
        try {
            audioTrack?.stop()
            audioTrack?.flush()
        } catch (_: IllegalStateException) {}
    }

    fun release() {
        try {
            audioTrack?.release()
        } catch (_: Exception) {}
        audioTrack = null
    }
}
