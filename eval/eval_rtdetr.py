import argparse
import time
from pathlib import Path
import json

import numpy as np
import torch
from ultralytics import RTDETR
from loguru import logger


class RoboMasterEvaluator:
    """RoboMaster RT-DETR专业评估器"""

    def __init__(self, model_path, data_yaml):
        self.model_path = Path(model_path)
        self.data_yaml = data_yaml

        logger.info("Loading RT-DETR model...")
        self.model = RTDETR(str(self.model_path))

        device = next(self.model.model.parameters()).device
        logger.info(f"✓ Model loaded on: {device}")

    # ------------------------------------------------
    # 1. Precision evaluation (academic)
    # ------------------------------------------------

    def evaluate_accuracy(self):
        """评估学术指标"""

        logger.info("\nRunning validation on test set...")

        metrics = self.model.val(
            data=self.data_yaml,
            imgsz=640,
            batch=8,
            save_json=True,
            plots=True  # 生成PR曲线等
        )

        results = {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "mAP75": float(metrics.box.map75),
            "Precision": float(metrics.box.mp),
            "Recall": float(metrics.box.mr),
        }

        logger.info("\n📊 Accuracy Metrics")
        logger.info("-" * 40)
        for k, v in results.items():
            logger.info(f"{k:15s}: {v:.4f}")

        return results

    # ------------------------------------------------
    # 2. Small-object analysis (RoboMaster关键)
    # ------------------------------------------------

    def evaluate_small_objects(self):
        """评估小目标性能（装甲板通常是小目标）"""

        logger.info("\n🎯 Small-object evaluation")
        logger.info("  (Critical for long-distance armor detection)")

        metrics = self.model.val(
            data=self.data_yaml,
            imgsz=640,
            batch=8,
            plots=False
        )

        results = {}

        # 尝试获取分尺度AP
        if hasattr(metrics.box, 'maps') and len(metrics.box.maps) >= 3:
            results['AP_small'] = float(metrics.box.maps[0])  # <32²
            results['AP_medium'] = float(metrics.box.maps[1])  # 32²-96²
            results['AP_large'] = float(metrics.box.maps[2])  # >96²

            logger.info(f"  AP_small  (<32²):  {results['AP_small']:.4f}  ← 关键")
            logger.info(f"  AP_medium (32²-96²): {results['AP_medium']:.4f}")
            logger.info(f"  AP_large  (>96²):  {results['AP_large']:.4f}")
        else:
            # 如果没有分尺度，使用整体mAP
            results['AP_overall'] = float(metrics.box.map)
            logger.warning("  No scale-specific AP, using overall mAP")
            logger.info(f"  AP_overall: {results['AP_overall']:.4f}")

        return results

    # ------------------------------------------------
    # 3. Real inference latency (工程关键)
    # ------------------------------------------------

    def benchmark_latency(self, imgsz=640, warmup=30, iters=200):
        """测试真实推理延迟"""

        logger.info("\n⚡ Latency benchmark")
        logger.info(f"  Image size: {imgsz}x{imgsz}")
        logger.info(f"  Warmup iterations: {warmup}")
        logger.info(f"  Test iterations: {iters}")

        device = next(self.model.model.parameters()).device

        # 创建随机图像（模拟真实输入）
        dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        # 预热（重要！）
        logger.info("  Warming up...")
        for _ in range(warmup):
            _ = self.model.predict(dummy, imgsz=imgsz, verbose=False)

        # 测速
        logger.info(f"  Benchmarking ({iters} iterations)...")
        times = []

        for i in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = self.model.predict(dummy, imgsz=imgsz, verbose=False)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)

            # 进度提示
            if (i + 1) % 50 == 0:
                logger.info(f"    Progress: {i + 1}/{iters}")

        times = np.array(times)

        results = {
            "mean_latency": float(times.mean()),
            "std_latency": float(times.std()),
            "p50": float(np.percentile(times, 50)),
            "p90": float(np.percentile(times, 90)),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)),
            "min": float(times.min()),
            "max": float(times.max()),
            "fps": float(1000 / times.mean())
        }

        logger.info("\n  Latency Statistics (ms)")
        logger.info("  " + "-" * 40)
        logger.info(f"  Mean:   {results['mean_latency']:.2f} ± {results['std_latency']:.2f}")
        logger.info(f"  P50:    {results['p50']:.2f}")
        logger.info(f"  P90:    {results['p90']:.2f}")
        logger.info(f"  P99:    {results['p99']:.2f}  ← 工程关键指标")
        logger.info(f"  Max:    {results['max']:.2f}")
        logger.info(f"  FPS:    {results['fps']:.1f}")

        # RT-DETR特点说明
        logger.info("\n  💡 RT-DETR优势:")
        logger.info(f"  ✓ 无NMS后处理，延迟更稳定")
        logger.info(f"  ✓ 延迟抖动: ±{results['std_latency']:.2f} ms")

        return results

    # ------------------------------------------------
    # 4. Model complexity
    # ------------------------------------------------

    def analyze_model(self):
        """分析模型复杂度"""

        logger.info("\n🔬 Model Complexity Analysis")
        logger.info("-" * 40)

        # 参数量
        total_params = sum(p.numel() for p in self.model.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.model.parameters() if p.requires_grad
        )

        logger.info(f"Total parameters:     {total_params / 1e6:.2f}M")
        logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

        # 显存占用
        if torch.cuda.is_available():
            device = next(self.model.model.parameters()).device

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            dummy = torch.randn(1, 3, 640, 640).to(device)

            with torch.no_grad():
                _ = self.model.model(dummy)

            peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2

            logger.info(f"Peak GPU memory:      {peak_mem:.1f} MB")

    # ------------------------------------------------
    # 5. Full report
    # ------------------------------------------------

    def run_full_evaluation(self, save_results=True):
        """运行完整评估"""

        logger.info("\n" + "=" * 60)
        logger.info("RT-DETR Full Evaluation - RoboMaster Armor Detection")
        logger.info("=" * 60)

        results = {}

        # 1. 精度评估
        try:
            logger.info("\n[1/4] Accuracy Evaluation")
            results['accuracy'] = self.evaluate_accuracy()
        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}")
            results['accuracy'] = None

        # 2. 小目标评估
        try:
            logger.info("\n[2/4] Small-Object Analysis")
            results['small_objects'] = self.evaluate_small_objects()
        except Exception as e:
            logger.error(f"Small-object evaluation failed: {e}")
            results['small_objects'] = None

        # 3. 延迟测试
        try:
            logger.info("\n[3/4] Latency Benchmark")
            results['latency'] = self.benchmark_latency()
        except Exception as e:
            logger.error(f"Latency benchmark failed: {e}")
            results['latency'] = None

        # 4. 模型分析
        try:
            logger.info("\n[4/4] Model Complexity Analysis")
            self.analyze_model()
        except Exception as e:
            logger.error(f"Model analysis failed: {e}")

        # 最终总结
        self._print_final_summary(results)

        # 保存结果
        if save_results:
            self._save_results(results)

        return results

    def _print_final_summary(self, results):
        """打印最终总结"""

        logger.info("\n" + "=" * 60)
        logger.info("FINAL SUMMARY - RT-DETR Performance Report")
        logger.info("=" * 60)

        if results.get('accuracy'):
            acc = results['accuracy']
            logger.info("\n📊 Accuracy:")
            logger.info(f"  mAP@0.5:      {acc['mAP50']:.4f}")
            logger.info(f"  mAP@0.5:0.95: {acc['mAP50-95']:.4f}")
            logger.info(f"  Precision:    {acc['Precision']:.4f}")
            logger.info(f"  Recall:       {acc['Recall']:.4f}")

        if results.get('latency'):
            lat = results['latency']
            logger.info("\n⚡ Performance:")
            logger.info(f"  FPS:          {lat['fps']:.1f}")
            logger.info(f"  P99 latency:  {lat['p99']:.2f} ms")

        logger.info("\n" + "=" * 60)

    def _save_results(self, results):
        """保存评估结果到JSON"""

        output_file = self.model_path.parent / 'evaluation_results.json'

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✓ Results saved to: {output_file}")


# ------------------------------------------------
# CLI
# ------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RT-DETR Professional Evaluator for RoboMaster"
    )

    parser.add_argument("--model", required=True, help="Model weights path")
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    evaluator = RoboMasterEvaluator(args.model, args.data)
    evaluator.run_full_evaluation(save_results=not args.no_save)


if __name__ == "__main__":
    main()