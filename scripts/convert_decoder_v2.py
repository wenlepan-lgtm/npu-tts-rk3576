#!/usr/bin/env python3
"""
Piper streaming decoder ONNX -> RKNN 转换脚本（自动检测版）

与 convert_decoder.py 功能相同，但不手动指定输入形状，
让 RKNN Toolkit2 自动从 ONNX 图中推断。

适用场景:
    - 不确定 decoder.onnx 输入形状时
    - 不同版本的 Piper 模型
"""

from rknn.api import RKNN
import os


def main():
    rknn = RKNN()

    print('Configuring RKNN model for RK3576...')
    rknn.config(
        target_platform='RK3576',
        optimization_level=3
    )

    print('Loading ONNX model...')
    onnx_path = 'decoder.onnx'
    if not os.path.exists(onnx_path):
        print(f'Error: {onnx_path} not found!')
        return -1

    ret = rknn.load_onnx(model=onnx_path)
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

    output_path = 'decoder_rk3576_v2.rknn'
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
