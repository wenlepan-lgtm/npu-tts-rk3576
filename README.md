# npu-tts-rk3576

在 RK3576 Android 设备上跑中文 TTS，decoder 走 NPU 加速。

实测性能：冷启动 ~500ms，合成 ~30ms/char。

## 它能干什么

输入中文文本，输出语音。全程离线，不联网。encoder 跑 CPU（ONNX Runtime），decoder 跑 NPU（RKNN Runtime）。

```
中文文本 → 拼音转音素 → encoder.onnx (CPU) → decoder.rknn (NPU) → 音频播放
```

## 从零开始：完整搭建指南

下面每一步都可以直接复制执行。按顺序走完就能跑起来。

### 第 1 步：Clone 仓库

```bash
git clone https://github.com/wenlepan-lgtm/npu-tts-rk3576.git
cd npu-tts-rk3576
```

### 第 2 步：下载 ONNX Runtime

需要 `.so` 文件（Android 运行时）和头文件（C++ 编译用）。

```bash
# 下载 ONNX Runtime 1.17.1 for Android
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-android-1.17.1.zip
unzip onnxruntime-android-1.17.1.zip -d onnxruntime-android

# 复制 .so 到 jniLibs
mkdir -p android/app/src/main/jniLibs/arm64-v8a
cp onnxruntime-android/jni/arm64-v8a/libonnxruntime.so \
   android/app/src/main/jniLibs/arm64-v8a/

# 复制头文件
mkdir -p android/app/src/main/cpp/onnxruntime/include
cp onnxruntime-android/headers/* \
   android/app/src/main/cpp/onnxruntime/include/

cd -
```

**验证：**
```bash
ls android/app/src/main/jniLibs/arm64-v8a/libonnxruntime.so   # 应该存在，~60MB
ls android/app/src/main/cpp/onnxruntime/include/onnxruntime_cxx_api.h  # 应该存在
```

### 第 3 步：下载 RKNN Runtime

```bash
cd /tmp

# 克隆 RKNPU2 仓库（包含 runtime .so 和头文件）
git clone --depth 1 https://github.com/rockchip-linux/rknpu2.git

# 复制 .so 到 jniLibs
cp rknpu2/runtime/Android/librknn_api/arm64-v8a/librknnrt.so \
   android/app/src/main/jniLibs/arm64-v8a/

# 复制头文件
mkdir -p android/app/src/main/cpp/rknn
cp rknpu2/runtime/Android/librknn_api/include/rknn_api.h \
   android/app/src/main/cpp/rknn/

cd -
```

**验证：**
```bash
ls android/app/src/main/jniLibs/arm64-v8a/librknnrt.so      # 应该存在，~7MB
ls android/app/src/main/cpp/rknn/rknn_api.h                   # 应该存在
```

### 第 4 步：准备模型文件

这一步需要一个 **x86 Linux** 机器（Ubuntu 20.04+），因为 RKNN Toolkit2 只支持 x86。

如果你的模型已经转好了，直接跳到推送设备的步骤。

#### 4a. 安装 RKNN Toolkit2

在 x86 Linux 上：

```bash
pip install rknn-toolkit2==1.6.0
# 或从 GitHub 安装具体 whl：
# pip install https://github.com/airockchip/rknn-toolkit2/raw/master/rknn-toolkit2/packages/rknn_toolkit2-1.6.0+81f21f4d-cp310-cp310-linux_x86_64.whl
```

#### 4b. 转换 decoder

把你的 `decoder.onnx` 转成 RKNN 格式：

```bash
cd scripts
python3 piper_decoder2rknn.py /path/to/decoder.onnx decoder_rk3576.rknn --platform rk3576
```

转换需要几分钟，输出 `decoder_rk3576.rknn`（约 47MB）。

**验证：**
```bash
ls -lh decoder_rk3576.rknn  # 应该存在，~47MB
```

#### 4c. 推送模型到 Android 设备

```bash
adb shell mkdir -p /sdcard/npu-tts-models
adb push /path/to/encoder.onnx /sdcard/npu-tts-models/
adb push decoder_rk3576.rknn /sdcard/npu-tts-models/
# 如果有 speaker embedding：
adb push /path/to/g.bin /sdcard/npu-tts-models/
```

**验证：**
```bash
adb shell ls -l /sdcard/npu-tts-models/
# 应该看到 encoder.onnx 和 decoder_rk3576.rknn
```

### 第 5 步：构建 Android App

```bash
cd android

# 生成 Gradle Wrapper（首次需要）
gradle wrapper --gradle-version 8.2

# 构建 Debug APK
./gradlew assembleDebug
```

也可以用 Android Studio：打开 `android/` 目录，等 Gradle Sync 完成，点 Run。

**注意：** 只编译 `arm64-v8a`，因为 RK3576 是 aarch64。

### 第 6 步：安装并运行

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.nputts.rk3576/.MainActivity
```

App 界面：
1. 等状态显示 "Engine: Ready"
2. 输入中文文本
3. 点 "Synthesize (NPU)"
4. 听到语音

---

## 集成到你自己的 App

如果不用 Demo App，只想用 NPU TTS 引擎：

### 1. 复制文件

把以下文件复制到你的 Android 项目：

```
cpp/            → app/src/main/cpp/
jniLibs/        → app/src/main/jniLibs/
NpuTtsEngine.kt
NpuTtsNative.kt
PinyinToPhoneme.kt
AudioPlayer.kt
```

### 2. 调用

```kotlin
val engine = NpuTtsEngine()
engine.init("/sdcard/npu-tts-models")  // 模型目录

// 同步调用（建议在 IO 线程）
val pcm: FloatArray? = engine.synthesize("你好世界")
// pcm 是 44100Hz mono float32 PCM 数据

// 用 AudioTrack 播放
val player = AudioPlayer()
player.init()
player.play(pcm!!)

// 用完释放
engine.release()
player.release()
```

### 3. Gradle 配置

在 `app/build.gradle.kts` 中添加：

```kotlin
android {
    defaultConfig {
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += listOf("-DANDROID_STL=c++_shared")
            }
        }
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }
}
```

---

## 文件说明

| 文件 | 作用 |
|------|------|
| `scripts/piper_decoder2rknn.py` | 完整版 ONNX→RKNN 转换，支持分析/验证 |
| `scripts/convert_decoder.py` | 简化版转换（显式输入形状） |
| `scripts/convert_decoder_v2.py` | 简化版转换（自动检测形状） |
| `android/app/src/main/cpp/npu_tts_jni.cpp` | JNI 入口：init/synthesize/release |
| `android/app/src/main/cpp/rknn_decoder.cpp` | RKNN NPU decoder 封装 |
| `android/app/src/main/cpp/onnx_encoder.cpp` | ONNX Runtime CPU encoder 封装 |
| `android/app/src/main/cpp/audio_utils.cpp` | 音频归一化（解决 RKNN 输出 -30dB 问题） |
| `android/app/src/main/cpp/rknn_stub.c` | 非 RKNN 设备上的编译存根 |
| `android/app/src/main/java/.../NpuTtsEngine.kt` | 核心 TTS 引擎 |
| `android/app/src/main/java/.../PinyinToPhoneme.kt` | 中文→音素映射（20920 个 CJK 字符） |

## 常见问题

**Q: App 启动后显示 "Init failed"**

检查模型文件是否在正确位置：
```bash
adb shell ls -l /sdcard/npu-tts-models/
# 必须有 encoder.onnx 和 decoder_rk3576.rknn
```

**Q: 编译报错找不到 `onnxruntime_cxx_api.h`**

第 2 步没做。下载 ONNX Runtime 并复制头文件到 `cpp/onnxruntime/include/`。

**Q: 编译报错找不到 `rknn_api.h`**

第 3 步没做。下载 RKNPU2 并复制头文件到 `cpp/rknn/`。如果暂时没有，编译会自动使用 stub（NPU 不可用但能编译通过）。

**Q: 合成返回 null**

- 检查 logcat 中 `NpuTtsEngine` 和 `NPU TTS` tag 的错误日志
- 确认 encoder.onnx 和 decoder.rk3576.rknn 匹配（同一个模型拆出来的）
- 确认设备是 RK3576（`adb shell cat /proc/cpuinfo | grep Hardware`）

**Q: 音频声音很小**

正常现象。RKNN decoder 输出幅度约 -30dB，引擎已内置自动归一化。如果仍然很小，检查 `audio_normalize` 是否被调用。

**Q: 能不能在模拟器上测试**

不行。RKNN Runtime 需要真实的 Rockchip NPU 硬件。

## 已知限制

- 只支持 `arm64-v8a`（RK3576 架构）
- 中文模型需要单独获取 MeloTTS 或 Piper 的 ONNX 模型
- RKNN Toolkit2 模型转换只能在 x86 Linux 上进行
- decoder 使用固定输入形状（padded），RKNN 不支持动态 shape

## 相关项目

- [RKNN Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2) — ONNX→RKNN 模型转换
- [RKNPU2](https://github.com/rockchip-linux/rknpu2) — RKNN Runtime
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — CPU 推理引擎
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 语音推理框架（本项目未使用其 TTS NPU 部分）
- [Paroli](https://github.com/marty1885/paroli) — RK3588 上的 Piper NPU TTS（C++ Linux，非 Android）

## License

MIT
