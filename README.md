# npu-tts-rk3576

Offline Chinese TTS on RK3576 Android, accelerated by NPU.

Encoder runs on CPU (ONNX Runtime), decoder runs on NPU (RKNN Runtime). The decoder is the heaviest computation, so offloading it to NPU gives significant speedup.

**Performance (measured on RK3576):** cold start ~500ms, synthesis ~30ms/char.

## Architecture

```
中文文本 → PinyinToPhoneme → phones/tones/langs
         → [encoder.onnx / CPU] → latent (z, y_mask)
         → [decoder.rknn / NPU] → PCM float32 audio
         → normalize → AudioTrack playback
```

## Quick Start

### 1. Prepare native libraries

Download and place in `android/app/src/main/jniLibs/arm64-v8a/`:

| Library | Source |
|---------|--------|
| `libonnxruntime.so` | [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) |
| `librknnrt.so` | [RKNPU2 runtime](https://github.com/rockchip-linux/rknpu2) |

Download and place headers:
- `android/app/src/main/cpp/rknn/rknn_api.h` from RKNPU2
- `android/app/src/main/cpp/onnxruntime/include/` from ONNX Runtime

See individual README.md files in those directories for details.

### 2. Prepare models

Convert your TTS model decoder to RKNN format on x86 Linux:

```bash
pip install rknn-toolkit2>=1.6.0
python scripts/piper_decoder2rknn.py decoder.onnx decoder_rk3576.rknn --platform rk3576
```

Or use the simpler scripts:

```bash
python scripts/convert_decoder.py   # explicit input shapes
python scripts/convert_decoder_v2.py # auto-detect shapes
```

### 3. Push models to device

```bash
adb shell mkdir -p /sdcard/npu-tts-models
adb push encoder.onnx /sdcard/npu-tts-models/
adb push decoder_rk3576.rknn /sdcard/npu-tts-models/
adb push g.bin /sdcard/npu-tts-models/  # optional: speaker embedding
```

### 4. Build and run

Open `android/` in Android Studio, build, and deploy to your RK3576 device.

Or from command line:

```bash
cd android
./gradlew installDebug
```

Type Chinese text, press Synthesize, hear NPU-accelerated TTS.

## Project Structure

```
scripts/                         # Model conversion tools
├── piper_decoder2rknn.py       # Full-featured ONNX → RKNN converter
├── convert_decoder.py           # Simple converter (explicit shapes)
├── convert_decoder_v2.py        # Simple converter (auto-detect)
└── prepare_models.sh            # One-click model preparation

android/                         # Android demo app
├── app/src/main/
│   ├── cpp/                     # C++ JNI layer
│   │   ├── npu_tts_jni.cpp     #   JNI entry point
│   │   ├── rknn_decoder.cpp    #   RKNN NPU decoder wrapper
│   │   ├── onnx_encoder.cpp    #   ONNX Runtime CPU encoder wrapper
│   │   ├── audio_utils.cpp     #   Audio normalization
│   │   ├── rknn_stub.c         #   Build stubs for non-RKNN devices
│   │   └── CMakeLists.txt
│   ├── java/.../
│   │   ├── NpuTtsEngine.kt     #   Core TTS engine
│   │   ├── NpuTtsNative.kt     #   JNI declarations
│   │   ├── PinyinToPhoneme.kt  #   Chinese text → phoneme mapping (20920 CJK chars)
│   │   ├── AudioPlayer.kt      #   AudioTrack playback
│   │   └── MainActivity.kt     #   Demo UI (Compose)
│   └── jniLibs/                 # Native .so files (not in git)
```

## Supported Models

| Model | Language | Encoder | Decoder | Notes |
|-------|----------|---------|---------|-------|
| MeloTTS | Chinese (cmn) | CPU (ONNX) | NPU (RKNN) | Primary target |
| Piper VITS | English (en-us) | CPU (ONNX) | NPU (RKNN) | Streaming split required |

### Model files needed

```
/sdcard/npu-tts-models/
├── encoder.onnx          # CPU inference
├── decoder_rk3576.rknn   # NPU inference
└── g.bin                 # Speaker embedding (optional)
```

## Integration into your own app

### Kotlin

```kotlin
val engine = NpuTtsEngine()
engine.init("/sdcard/npu-tts-models")

val pcm: FloatArray? = engine.synthesize("你好世界")
// pcm: 44100Hz mono float32 samples

engine.release()
```

### C++ (standalone, without Android)

Use `rknn_decoder.h` and `onnx_encoder.h` directly:

```cpp
OnnxEncoder* enc = encoder_init("encoder.onnx", "g.bin");
RknnDecoder* dec = rknn_decoder_init("decoder_rk3576.rknn");

EncoderOutput out;
encoder_run(enc, phones, tones, langs, len, 0.3f, 1.0f, &out);

float* audio;
int audio_len;
rknn_decoder_run(dec, out.z, out.z_len, out.y_mask, out.mask_len, &audio, &audio_len);
```

## Known Issues

- **Audio normalization**: RKNN decoder output has ~-30dB amplitude. The engine auto-normalizes to 0.9 peak.
- **Fixed input shapes**: RKNN requires fixed tensor shapes. The decoder uses padded input.
- **RKNN Toolkit2**: Must run on x86 Linux. Not available on macOS/ARM.

## Requirements

- RK3576 Android device (arm64-v8a)
- Android Studio with NDK 25+
- x86 Linux for model conversion (RKNN Toolkit2)

## References

- [RKNN Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2)
- [RKNPU2 Runtime](https://github.com/rockchip-linux/rknpu2)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [Piper TTS](https://github.com/rhasspy/piper)

## License

MIT
