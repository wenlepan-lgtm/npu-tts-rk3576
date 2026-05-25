# Model Preparation

## Required files

Place these on the Android device at `/sdcard/npu-tts-models/`:

| File | Size | Purpose |
|------|------|---------|
| `encoder.onnx` | ~30 MB | MeloTTS/VITS encoder, runs on CPU |
| `decoder_rk3576.rknn` | ~47 MB | Decoder, runs on NPU |
| `g.bin` | ~1 MB | Speaker embedding (optional) |

## How to convert

### Step 1: Get the ONNX model

For MeloTTS, use the official export tool to get encoder.onnx and decoder.onnx.

For Piper/VITS, use the [mush42/piper streaming branch](https://github.com/mush42/piper):
```bash
python3 -m piper_train.export_onnx_streaming /path/to/checkpoint /output/dir
```

### Step 2: Convert decoder to RKNN

Requires x86 Linux with RKNN Toolkit2:
```bash
pip install rknn-toolkit2>=1.6.0
python scripts/piper_decoder2rknn.py decoder.onnx decoder_rk3576.rknn --platform rk3576
```

### Step 3: Deploy to device

```bash
adb shell mkdir -p /sdcard/npu-tts-models
adb push encoder.onnx decoder_rk3576.rknn g.bin /sdcard/npu-tts-models/
```

## Pre-built models

Pre-converted models may be available on the [GitHub Releases](https://github.com/wenlepan-lgtm/npu-tts-rk3576/releases) page.
