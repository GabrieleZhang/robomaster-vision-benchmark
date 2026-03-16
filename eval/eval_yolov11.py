import argparse
from pathlib import Path
import torch
import time
import numpy as np
from loguru import logger

from ultralytics import YOLO


class YOLO11Evaluator:
    """YOLOv11评估器"""

    def __init__(self, model_path, data_yaml=None):
        """
        初始化评估器

        Args:
            model_path: 模型权重路径
            data_yaml: 数据集配置文件路径
        """
        self.model_path = Path(model_path)
        self.data_yaml = data_yaml

        if not self.model_path.exists():
            logger.error(f"模型文件不存在: {self.model_path}")
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info("=" * 60)
        logger.info("YOLOv11 模型评估")
        logger.info("=" * 60)
        logger.info(f"模型路径: {self.model_path}")
        if self.data_yaml:
            logger.info(f"数据集: {self.data_yaml}")
        logger.info("=" * 60)

        # 加载模型
        self.load_model()

    def load_model(self):
        """加载模型"""
        logger.info("\n加载模型...")

        try:
            self.model = YOLO(self.model_path)

            # 获取模型信息
            device = next(self.model.model.parameters()).device
            logger.info(f"✓ 模型加载成功")
            logger.info(f"  设备: {device}")
            logger.info(f"  模型类型: YOLOv11")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def evaluate_accuracy(self):
        """评估模型精度"""

        if not self.data_yaml:
            logger.warning("未提供数据集配置，跳过精度评估")
            return None

        logger.info("\n" + "=" * 60)
        logger.info("评估模型精度")
        logger.info("=" * 60)

        try:
            # 运行验证
            logger.info("开始验证集评估...")
            metrics = self.model.val(data=self.data_yaml)

            # 提取关键指标
            results = {
                'mAP@0.5': float(metrics.box.map50),
                'mAP@0.5:0.95': float(metrics.box.map),
                'mAP@0.75': float(metrics.box.map75),
                'Precision': float(metrics.box.mp),
                'Recall': float(metrics.box.mr),
            }

            # 打印结果
            logger.info("\n精度指标:")
            logger.info("-" * 40)
            for key, value in results.items():
                logger.info(f"  {key:20s}: {value:.4f}")
            logger.info("-" * 40)

            return results

        except Exception as e:
            logger.error(f"精度评估失败: {e}")
            return None

    def measure_speed(self, num_warmup=10, num_iterations=100, imgsz=640):
        """
        测量推理速度

        Args:
            num_warmup: 预热次数
            num_iterations: 测试迭代次数
            imgsz: 输入图像尺寸
        """
        logger.info("\n" + "=" * 60)
        logger.info("测量推理速度")
        logger.info("=" * 60)
        logger.info(f"预热次数: {num_warmup}")
        logger.info(f"测试次数: {num_iterations}")
        logger.info(f"图像尺寸: {imgsz}x{imgsz}")

        device = next(self.model.model.parameters()).device

        # 创建随机输入
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)

        # 预热
        logger.info("\n预热中...")
        self.model.model.eval()
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model.model(dummy_input)

        # 测速
        logger.info(f"测量中 ({num_iterations} 次)...")
        times = []

        with torch.no_grad():
            for i in range(num_iterations):
                if device.type == 'cuda':
                    torch.cuda.synchronize()

                start = time.time()
                _ = self.model.model(dummy_input)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                times.append(time.time() - start)

                # 进度提示
                if (i + 1) % 20 == 0:
                    logger.info(f"  进度: {i + 1}/{num_iterations}")

        # 统计分析
        times = np.array(times) * 1000  # 转换为ms

        results = {
            'mean_ms': float(times.mean()),
            'std_ms': float(times.std()),
            'min_ms': float(times.min()),
            'max_ms': float(times.max()),
            'p50': float(np.percentile(times, 50)),
            'p90': float(np.percentile(times, 90)),
            'p99': float(np.percentile(times, 99)),
            'fps': float(1000 / times.mean()),
        }

        # 打印结果
        logger.info("\n速度测试结果:")
        logger.info("-" * 40)
        logger.info(f"  平均推理时间: {results['mean_ms']:.2f} ms")
        logger.info(f"  标准差:       {results['std_ms']:.2f} ms")
        logger.info(f"  最小时间:     {results['min_ms']:.2f} ms")
        logger.info(f"  最大时间:     {results['max_ms']:.2f} ms")
        logger.info(f"  P99延迟:      {results['p99']:.2f} ms")
        logger.info(f"  FPS:          {results['fps']:.1f}")
        logger.info("-" * 40)

        # YOLOv11特点说明
        logger.info("\n💡 YOLOv11特点:")
        logger.info("  ✓ C3k2 + C2PSA架构，精度更高")
        logger.info("  ✓ 需要NMS后处理")
        logger.info(f"  ✓ 延迟抖动: ±{results['std_ms']:.2f} ms")

        return results

    def test_nms_impact(self, imgsz=640):
        """
        测试NMS对延迟的影响
        """
        logger.info("\n" + "=" * 60)
        logger.info("NMS延迟影响分析")
        logger.info("=" * 60)

        device = next(self.model.model.parameters()).device
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)

        # 测试纯模型推理（不含NMS）
        times_no_nms = []
        self.model.model.eval()
        with torch.no_grad():
            for _ in range(50):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                _ = self.model.model(dummy_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times_no_nms.append((time.time() - start) * 1000)

        times_no_nms = np.array(times_no_nms)

        logger.info("\n纯模型推理（不含NMS）:")
        logger.info(f"  平均延迟: {times_no_nms.mean():.2f} ms")
        logger.info(f"  延迟抖动: ±{times_no_nms.std():.2f} ms")

        # 测试完整推理（含NMS）
        dummy_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        times_with_nms = []

        for _ in range(50):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _ = self.model.predict(dummy_img, imgsz=imgsz, verbose=False)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times_with_nms.append((time.time() - start) * 1000)

        times_with_nms = np.array(times_with_nms)

        logger.info("\n完整推理（含NMS）:")
        logger.info(f"  平均延迟: {times_with_nms.mean():.2f} ms")
        logger.info(f"  延迟抖动: ±{times_with_nms.std():.2f} ms")

        # NMS开销
        nms_overhead = times_with_nms.mean() - times_no_nms.mean()

        logger.info("\nNMS开销分析:")
        logger.info(f"  NMS额外延迟: {nms_overhead:.2f} ms")
        logger.info(f"  占总延迟比例: {(nms_overhead / times_with_nms.mean()) * 100:.1f}%")
        logger.info(f"  延迟抖动增加: ±{times_with_nms.std() - times_no_nms.std():.2f} ms")

        logger.info("\n💡 对比YOLO26/RT-DETR:")
        logger.info("  YOLOv11: 需要NMS后处理")
        logger.info("  YOLO26/RT-DETR: 端到端，无NMS")
        logger.info("  → 延迟更稳定，部署更简单")

    def analyze_model(self):
        """分析模型结构"""
        logger.info("\n" + "=" * 60)
        logger.info("模型结构分析")
        logger.info("=" * 60)

        # 统计参数量
        total_params = sum(p.numel() for p in self.model.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)

        logger.info(f"\n参数统计:")
        logger.info(f"  总参数量:   {total_params:,} ({total_params / 1e6:.1f}M)")
        logger.info(f"  可训练参数: {trainable_params:,} ({trainable_params / 1e6:.1f}M)")

        # 显存占用
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2
            mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2

            logger.info(f"\n显存占用:")
            logger.info(f"  已分配: {mem_allocated:.1f} MB")
            logger.info(f"  已预留: {mem_reserved:.1f} MB")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 完整评估（精度+速度）
    python eval_yolo11.py --model.../best.pt --data dataset.yaml

    # 仅速度测试
    python eval_yolo11.py --model .../best.pt --speed-test

    # NMS影响分析
    python eval_yolo11.py --model .../best.pt --test-nms
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='模型权重路径'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='数据集配置文件路径（用于精度评估）'
    )

    parser.add_argument(
        '--speed-test',
        action='store_true',
        help='进行速度测试'
    )

    parser.add_argument(
        '--test-nms',
        action='store_true',
        help='测试NMS延迟影响'
    )

    parser.add_argument(
        '--speed-iters',
        type=int,
        default=100,
        help='速度测试迭代次数'
    )

    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='输入图像尺寸'
    )

    parser.add_argument(
        '--analyze',
        action='store_true',
        help='分析模型结构'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建评估器
    evaluator = YOLO11Evaluator(args.model, args.data)

    # 精度评估
    if args.data:
        evaluator.evaluate_accuracy()

    # 速度测试
    if args.speed_test or not args.data:
        evaluator.measure_speed(
            num_iterations=args.speed_iters,
            imgsz=args.imgsz
        )

    # NMS影响测试
    if args.test_nms:
        evaluator.test_nms_impact(imgsz=args.imgsz)

    # 模型分析
    if args.analyze:
        evaluator.analyze_model()

    logger.info("\n 评估完成！")


if __name__ == "__main__":
    main()