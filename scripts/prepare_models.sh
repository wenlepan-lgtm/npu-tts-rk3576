#!/bin/bash
# 一键准备 NPU TTS 模型
# 用法: ./prepare_models.sh [output_dir]
#
# 前置条件:
#   - x86 Linux (Ubuntu 20.04+)
#   - Python 3.10
#   - rknn-toolkit2 >= 1.6.0
#   - pip install rknn-toolkit2 onnx onnxruntime
#
# 输出:
#   output_dir/
#   ├── encoder.onnx          # CPU 推理用
#   ├── decoder_rk3576.rknn   # NPU 推理用
#   └── g.bin                  # speaker embedding

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${1:-./npu-tts-models}"

echo "=== NPU TTS Model Preparation ==="
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# 检查依赖
check_deps() {
    python3 -c "from rknn.api import RKNN" 2>/dev/null || {
        echo "Error: rknn-toolkit2 not installed"
        echo "Install: pip install rknn_toolkit2-1.6.0+81f21f4d-cp310-cp310-linux_x86_64.whl"
        exit 1
    }
    echo "[OK] rknn-toolkit2"
}

# 如果提供了 encoder.onnx 和 decoder.onnx，转换为 RKNN
convert_decoder() {
    local decoder_onnx="$1"
    local target="$2"

    if [ ! -f "$decoder_onnx" ]; then
        echo "Error: decoder.onnx not found at $decoder_onnx"
        echo ""
        echo "To get the streaming-split model, use mush42/piper streaming branch:"
        echo "  git clone -b streaming https://github.com/mush42/piper.git"
        echo "  python3 -m piper_train.export_onnx_streaming <checkpoint> <output_dir>"
        echo ""
        echo "Or use pre-split models from HuggingFace if available."
        exit 1
    fi

    echo ""
    echo "--- Converting decoder to RKNN ---"
    python3 "$SCRIPT_DIR/piper_decoder2rknn.py" \
        "$decoder_onnx" \
        "$target" \
        --platform rk3576 \
        --optimization 3
}

check_deps

# 检查参数
if [ "$#" -ge 2 ]; then
    # 用户指定了 encoder.onnx 和 decoder.onnx 的位置
    DECODER_ONNX="$2"
    convert_decoder "$DECODER_ONNX" "$OUTPUT_DIR/decoder_rk3576.rknn"

    # 复制 encoder
    ENCODER_ONNX="${DECODER_ONnx%/*}/encoder.onnx"
    if [ -f "$ENCODER_ONNX" ]; then
        cp "$ENCODER_ONNX" "$OUTPUT_DIR/encoder.onnx"
        echo "[OK] Copied encoder.onnx"
    fi
else
    echo ""
    echo "Usage:"
    echo "  $0 <output_dir> <decoder.onnx>"
    echo ""
    echo "Example:"
    echo "  $0 ./models /path/to/decoder.onnx"
    echo ""
    echo "This will:"
    echo "  1. Convert decoder.onnx -> decoder_rk3576.rknn for NPU"
    echo "  2. Copy encoder.onnx for CPU inference"
    echo ""
    echo "Deploy to Android device:"
    echo "  adb push $OUTPUT_DIR/* /sdcard/npu-tts-models/"
fi

echo ""
echo "Done."
