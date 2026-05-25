#!/usr/bin/env python3
"""
Piper streaming decoder ONNX -> RKNN 转换脚本（显式输入形状版）

将 Piper streaming split 的 decoder.onnx 转换为 RK3576 NPU 可用的 .rknn 格式。
编码器（encoder）保持 CPU 推理，解码器（decoder）放到 NPU。

用法:
    1. 先从 mush42/piper streaming 分支导出 encoder.onnx + decoder.onnx
    2. 把 decoder.onnx 放在同级目录
    3. 在 x86 机器上运行本脚本（RKNN Toolkit2 限制）
    4. 产出 decoder_rk3576.rknn，推到 Android 设备

依赖:
    pip install rknn-toolkit2==1.6.0  (仅支持 x86 Linux + Python 3.8-3.10)

适用场景:
    - RK3576 Android 设备上的离线 TTS
    - Piper VITS 模型的 NPU 加速推理
    - 酒店 IPTV、智能音箱等嵌入式语音合成
"""

from rknn.api import RKNN
import os


def main():
    rknn = RKNN()

    print('Configuring RKNN model for RK3576...')
    rknn.config(
        target_platform='RK3576',
        optimization_level=3,
        single_core_mode=False
    )

    print('Loading ONNX model...')
    onnx_path = 'decoder.onnx'
    if not os.path.exists(onnx_path):
        print(f'Error: {onnx_path} not found!')
        return -1

    # Piper streaming decoder 的固定输入形状
    # z: [1, 192, 200] - latent vector
    # y_mask: [1, 1, 200] - mask
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=['z', 'y_mask'],
        input_size_list=[[1, 192, 200], [1, 1, 200]]
    )
    if ret != 0:
        print(f'Load ONNX failed! ret={ret}')
        return -1
    print('ONNX model loaded successfully')

    print('Building RKNN model (this may take several minutes)...')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f'Build RKNN failed! ret={ret}')
        return -1
    print('RKNN model built successfully')

    output_path = 'decoder_rk3576.rknn'
    print(f'Exporting RKNN model to {output_path}...')
    ret = rknn.export_rknn(export_path=output_path)
    if ret != 0:
        print(f'Export RKNN failed! ret={ret}')
        return -1

    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f'Success! Output: {output_path} ({size / 1024 / 1024:.1f} MB)')
    else:
        print(f'Error: Output file not created!')

    rknn.release()
    return 0


if __name__ == '__main__':
    exit(main())
