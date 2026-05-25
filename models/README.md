# 模型文件说明

## 需要什么文件

```
/sdcard/npu-tts-models/
├── encoder.onnx          # ~30MB, CPU 推理
├── decoder_rk3576.rknn   # ~47MB, NPU 推理
└── g.bin                 # ~1MB,  speaker embedding（可选）
```

## 怎么获取

### 方案 A：用现有的 MeloTTS 模型

如果你有 MeloTTS 训练好的模型 checkpoint，需要导出为 ONNX 并拆分：

```python
# 导出 encoder.onnx + decoder.onnx（取决于你的 MeloTTS 版本）
# 具体导出方式参考 MeloTTS 官方仓库
```

### 方案 B：用 Piper streaming split 模型

使用 [mush42/piper](https://github.com/mush42/piper) streaming 分支持有的模型：

```bash
git clone -b streaming https://github.com/mush42/piper.git
cd piper
python3 -m piper_train.export_onnx_streaming /path/to/checkpoint /output/dir
# 产出 encoder.onnx + decoder.onnx
```

### 方案 C：下载预转换模型

如果 GitHub Release 页面有预转换模型，直接下载：

```bash
# 示例（实际 URL 以 Release 页面为准）
wget https://github.com/wenlepan-lgtm/npu-tts-rk3576/releases/download/v1.0/encoder.onnx
wget https://github.com/wenlepan-lgtm/npu-tts-rk3576/releases/download/v1.0/decoder_rk3576.rknn
```

## 转换 decoder 为 RKNN

拿到 `decoder.onnx` 后，在 x86 Linux 上转换：

```bash
pip install rknn-toolkit2>=1.6.0
cd scripts/
python3 piper_decoder2rknn.py /path/to/decoder.onnx ../models/decoder_rk3576.rknn --platform rk3576
```

## 推送到设备

```bash
adb shell mkdir -p /sdcard/npu-tts-models
adb push encoder.onnx /sdcard/npu-tts-models/
adb push decoder_rk3576.rknn /sdcard/npu-tts-models/
adb push g.bin /sdcard/npu-tts-models/  # 如果有的话
```

验证：

```bash
adb shell ls -lh /sdcard/npu-tts-models/
```

## 支持的模型类型

| 模型 | 语言 | 格式 | 说明 |
|------|------|------|------|
| MeloTTS | 中文 | encoder.onnx + decoder.onnx + g.bin | 推荐，中文效果好 |
| Piper VITS (streaming split) | 英文/多语言 | encoder.onnx + decoder.onnx | 需 mush42/piper streaming 分支导出 |

**关键：** 普通的 Piper 单文件 `.onnx` 不能直接用，必须先做 streaming split 拆成 encoder + decoder。
