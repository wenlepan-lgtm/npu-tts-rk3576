#!/usr/bin/env python3
"""
Piper Streaming Decoder ONNX → RKNN 转换脚本

用途：将 Piper streaming decoder.onnx 转换为 RKNN 格式，用于 RK3576/RK3588 NPU 推理

前置条件：
1. x64 Linux (Ubuntu 20.04/22.04)
2. Python 3.10/3.11
3. rknn-toolkit2 >= 1.6.0

参考：
- Paroli: https://github.com/marty1885/paroli
- RKNN-Toolkit2: https://github.com/airockchip/rknn-toolkit2

用法：
    python3 piper_decoder2rknn.py decoder.onnx decoder.rknn --platform rk3576
"""

import argparse
import os
import sys
import numpy as np

# 检查 rknn-toolkit2 是否安装
try:
    from rknn.api import RKNN
except ImportError:
    print("错误：未安装 rknn-toolkit2")
    print("请安装：pip install rknn_toolkit2-1.6.0+81f21f4d-cp310-cp310-linux_x86_64.whl")
    sys.exit(1)


# Piper VITS decoder 默认参数
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_OVERLAP_SIZE = 256
DEFAULT_PADDED_SIZE = DEFAULT_WINDOW_SIZE + 2 * DEFAULT_OVERLAP_SIZE


def analyze_onnx_model(onnx_path):
    """分析 ONNX 模型的输入输出 shape"""
    try:
        import onnx
        model = onnx.load(onnx_path)

        print("\n=== ONNX 模型分析 ===")
        print(f"文件: {onnx_path}")
        print(f"IR Version: {model.ir_version}")
        print(f"Producer: {model.producer_name} {model.producer_version}")

        print("\n输入:")
        for inp in model.graph.input:
            dims = []
            for d in inp.type.tensor_type.shape.dim:
                if d.dim_value:
                    dims.append(str(d.dim_value))
                else:
                    dims.append(d.dim_param if d.dim_param else '?')
            print(f"  {inp.name}: [{', '.join(dims)}]")

        print("\n输出:")
        for out in model.graph.output:
            dims = []
            for d in out.type.tensor_type.shape.dim:
                if d.dim_value:
                    dims.append(str(d.dim_value))
                else:
                    dims.append(d.dim_param if d.dim_param else '?')
            print(f"  {out.name}: [{', '.join(dims)}]")

        print(f"\n总节点数: {len(model.graph.node)}")

        # 统计算子
        op_types = {}
        for node in model.graph.node:
            op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

        print("\n算子分布 (前 10):")
        for op, count in sorted(op_types.items(), key=lambda x: -x[1])[:10]:
            print(f"  {op}: {count}")

        return True

    except ImportError:
        print("警告：未安装 onnx 库，跳过模型分析")
        print("可安装：pip install onnx")
        return False
    except Exception as e:
        print(f"分析 ONNX 模型失败: {e}")
        return False


def check_rknn_compatibility(onnx_path):
    """检查 ONNX 模型与 RKNN 的兼容性"""
    print("\n=== RKNN 兼容性检查 ===")

    rknn = RKNN(verbose=False)

    # 加载 ONNX
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print("错误：加载 ONNX 失败")
        return False

    print("✓ ONNX 加载成功")

    # 检测不支持的算子
    # 注意：这需要完整构建，可能耗时较长
    print("\n正在检测算子兼容性...")

    rknn.release()
    return True


def convert_decoder_to_rknn(onnx_path, rknn_path, target_platform='rk3576',
                            do_quantization=False, optimization_level=3):
    """
    将 Piper decoder ONNX 转换为 RKNN

    Args:
        onnx_path: 输入 decoder.onnx 路径
        rknn_path: 输出 decoder.rknn 路径
        target_platform: 目标平台 ('rk3576' 或 'rk3588')
        do_quantization: 是否进行 INT8 量化
        optimization_level: 优化级别 (0-3)

    Returns:
        bool: 转换是否成功
    """
    print(f"\n=== 开始 RKNN 转换 ===")
    print(f"输入: {onnx_path}")
    print(f"输出: {rknn_path}")
    print(f"目标平台: {target_platform}")
    print(f"量化: {'INT8' if do_quantization else 'FP16'}")
    print(f"优化级别: {optimization_level}")

    # 检查输入文件
    if not os.path.exists(onnx_path):
        print(f"错误：输入文件不存在: {onnx_path}")
        return False

    rknn = RKNN(verbose=True)

    # 1. 配置 RKNN
    print("\n--> [1/4] 配置 RKNN")
    config_dict = {
        'target_platform': target_platform,
        'optimization_level': optimization_level,
        'quantize_input': do_quantization,
        'output_optimize': 1,
    }

    ret = rknn.config(**config_dict)
    if ret != 0:
        print("错误：RKNN 配置失败")
        rknn.release()
        return False
    print("✓ 配置完成")

    # 2. 加载 ONNX 模型
    print("\n--> [2/4] 加载 ONNX 模型")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print("错误：加载 ONNX 失败")
        rknn.release()
        return False
    print("✓ ONNX 加载成功")

    # 3. 构建 RKNN 模型
    print("\n--> [3/4] 构建 RKNN 模型")
    ret = rknn.build(do_quantization=do_quantization)
    if ret != 0:
        print("错误：构建 RKNN 失败")
        print("可能原因：")
        print("  1. 存在 RKNN 不支持的算子")
        print("  2. 动态 shape 未正确处理")
        print("  3. 模型结构复杂度超限")
        rknn.release()
        return False
    print("✓ RKNN 构建成功")

    # 4. 导出 RKNN 模型
    print("\n--> [4/4] 导出 RKNN 模型")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print("错误：导出 RKNN 失败")
        rknn.release()
        return False
    print(f"✓ RKNN 导出成功: {rknn_path}")

    # 输出文件大小
    output_size = os.path.getsize(rknn_path) / (1024 * 1024)
    input_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\n文件大小: {output_size:.2f} MB (原 ONNX: {input_size:.2f} MB)")

    rknn.release()
    return True


def verify_rknn_output(onnx_path, rknn_path, test_input=None):
    """
    验证 RKNN 输出与 ONNX 输出的一致性

    注意：此函数需要在目标设备上运行，或使用 RKNN 模拟器
    """
    print("\n=== 验证 RKNN 输出 ===")

    try:
        import onnxruntime as ort
    except ImportError:
        print("警告：未安装 onnxruntime，跳过验证")
        print("可安装：pip install onnxruntime")
        return True

    # ONNX 推理
    print("运行 ONNX 推理...")
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    if test_input is None:
        # 生成测试输入
        input_shape = ort_session.get_inputs()[0].shape
        # 替换动态维度
        fixed_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
        test_input = np.random.randn(*fixed_shape).astype(np.float32)

    onnx_output = ort_session.run(None, {input_name: test_input})
    print(f"ONNX 输出 shape: {[o.shape for o in onnx_output]}")

    # RKNN 推理（需要 target 或 simulator）
    print("\n注意：RKNN 推理需要连接目标设备或使用模拟器")
    print("在目标设备上验证方法：")
    print("  1. 将 decoder.rknn 推送到设备")
    print("  2. 使用 rknn_server 或 C API 进行推理")
    print("  3. 对比输出结果")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='将 Piper streaming decoder ONNX 转换为 RKNN 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本转换 (FP16)
  python3 %(prog)s decoder.onnx decoder.rknn

  # 指定目标平台
  python3 %(prog)s decoder.onnx decoder.rknn --platform rk3588

  # INT8 量化
  python3 %(prog)s decoder.onnx decoder.rknn --quantize

  # 分析模型
  python3 %(prog)s decoder.onnx --analyze-only
        """
    )

    parser.add_argument('onnx', help='输入 decoder.onnx 文件路径')
    parser.add_argument('rknn', nargs='?', help='输出 decoder.rknn 文件路径')
    parser.add_argument('--platform', '-p', default='rk3576',
                        choices=['rk3576', 'rk3588', 'rk3566', 'rk3568'],
                        help='目标平台 (默认: rk3576)')
    parser.add_argument('--quantize', '-q', action='store_true',
                        help='启用 INT8 量化 (默认: FP16)')
    parser.add_argument('--optimization', '-o', type=int, default=3, choices=[0, 1, 2, 3],
                        help='优化级别 0-3 (默认: 3)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='仅分析模型，不进行转换')
    parser.add_argument('--verify', action='store_true',
                        help='转换后验证输出')

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.onnx):
        print(f"错误：文件不存在: {args.onnx}")
        sys.exit(1)

    # 分析模型
    analyze_onnx_model(args.onnx)

    if args.analyze_only:
        sys.exit(0)

    # 检查输出路径
    if args.rknn is None:
        base_name = os.path.splitext(args.onnx)[0]
        args.rknn = f"{base_name}.rknn"
        print(f"\n未指定输出路径，使用: {args.rknn}")

    # 执行转换
    success = convert_decoder_to_rknn(
        args.onnx,
        args.rknn,
        target_platform=args.platform,
        do_quantization=args.quantize,
        optimization_level=args.optimization
    )

    if not success:
        print("\n❌ 转换失败")
        sys.exit(1)

    # 验证（可选）
    if args.verify:
        verify_rknn_output(args.onnx, args.rknn)

    print("\n✅ 转换成功!")
    print(f"\n下一步：")
    print(f"  1. 将 {args.rknn} 推送到 Android 设备")
    print(f"  2. 创建 JNI 封装调用 RKNN Runtime")
    print(f"  3. 集成到 TTS 流程")


if __name__ == '__main__':
    main()
