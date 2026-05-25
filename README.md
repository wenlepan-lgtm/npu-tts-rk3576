# npu-tts-rk3576

Piper TTS decoder RKNN conversion for Rockchip RK3576 NPU.

## What this does

Converts the Piper streaming decoder (ONNX) to RKNN format for NPU-accelerated inference on RK3576 Android devices.

Architecture: encoder runs on CPU (ONNX Runtime), decoder runs on NPU (RKNN). The decoder is the heaviest part of Piper/VITS, so offloading it to NPU significantly reduces TTS latency.

## Use cases

- Offline TTS on RK3576 Android devices (smart TV, set-top box, smart speaker)
- Hotel IPTV voice assistant
- Any embedded Android scenario requiring low-latency Chinese/English TTS without cloud

## How to use

### 1. Prepare the ONNX model

Use [mush42/piper](https://github.com/mush42/piper) streaming branch to export encoder + decoder split:

```bash
python3 -m piper_train.export_onnx_streaming /path/to/checkpoint /output/dir
```

This produces `encoder.onnx` and `decoder.onnx`.

### 2. Convert decoder to RKNN

On an **x86 Linux** machine (RKNN Toolkit2 requirement):

```bash
pip install rknn-toolkit2==1.6.0
python convert_decoder.py
```

Output: `decoder_rk3576.rknn` (~47 MB)

### 3. Deploy to Android

- `encoder.onnx` -> CPU inference via ONNX Runtime
- `decoder_rk3576.rknn` -> NPU inference via RKNN Runtime
- `librknnrt.so` -> from [rknpu2](https://github.com/rockchip-linux/rknpu2)

## Files

| File | Description |
|------|-------------|
| `convert_decoder.py` | Conversion with explicit input shapes (recommended) |
| `convert_decoder_v2.py` | Conversion with auto-detected input shapes |

## Tested models

| Model | Language | Encoder | Decoder | Status |
|-------|----------|---------|---------|--------|
| Piper LJSpeech | English (en-us) | CPU (ONNX) | NPU (RKNN) | Converted |
| Piper zh_CN-huayan | Chinese (cmn) | CPU (ONNX) | NPU (RKNN) | Ready for conversion |

## References

- [Piper TTS](https://github.com/rhasspy/piper)
- [RKNN Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2)
- [RKNPU2 Runtime](https://github.com/rockchip-linux/rknpu2)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [Paroli (RK3588 reference)](https://github.com/marty1885/paroli)

## License

MIT
