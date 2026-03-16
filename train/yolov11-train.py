import argparse
import sys
from pathlib import Path
import yaml
import torch
from datetime import datetime
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("请先安装 Ultralytics: pip install ultralytics")
    sys.exit(1)


class YOLO11Trainer:
    """YOLOv11 训练器"""

    def __init__(self, config_path):
        """
        初始化训练器

        Args:
            config_path: YAML配置文件路径
        """
        self.config_path = Path(config_path)

        # 检查配置文件
        if not self.config_path.exists():
            logger.error(f"配置文件不存在: {self.config_path}")
            sys.exit(1)

        # 加载配置
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        logger.info("=" * 60)
        logger.info("YOLOv11 训练配置加载完成")
        logger.info("=" * 60)
        logger.info(f"配置文件: {self.config_path}")
        logger.info(f"模型类型: {self.config['model']['name']}")
        logger.info(f"数据集: {self.config['data']['yaml_path']}")
        logger.info("=" * 60)
        logger.info("\n💡 YOLOv11 特性:")
        logger.info("  ✓ C3k2模块 - 更高效的特征提取")
        logger.info("  ✓ C2PSA注意力机制 - 精度提升")
        logger.info("  ✓ 改进的PAN-FPN - 多尺度融合")
        logger.info("  ✓ Anchor-Free检测 - 部署友好")
        logger.info("=" * 60)

        # 验证配置
        self._validate_config()

    def _validate_config(self):
        """验证配置文件"""

        # 检查数据集路径
        data_yaml = Path(self.config['data']['yaml_path'])
        if not data_yaml.exists():
            logger.error(f"数据集配置文件不存在: {data_yaml}")
            sys.exit(1)

        # 检查CUDA可用性
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，将使用CPU训练（非常慢）")
        else:
            logger.info(f"CUDA设备: {torch.cuda.get_device_name(0)}")
            logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # 创建输出目录
        output_dir = Path(self.config['output']['project']) / self.config['output']['name']
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"输出目录: {output_dir}")

    def train(self):
        """启动训练"""

        logger.info("\n" + "=" * 60)
        logger.info("开始训练 YOLOv11")
        logger.info("=" * 60)

        # 记录开始时间
        start_time = datetime.now()
        logger.info(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # 加载模型
            model_name = self.config['model']['name']
            logger.info(f"\n加载模型: {model_name}")

            # 检查是否有预训练权重
            pretrained = self.config['model'].get('pretrained_weights', None)
            if pretrained and Path(pretrained).exists():
                logger.info(f"使用自定义权重: {pretrained}")
                model = YOLO(pretrained)
            else:
                logger.info(f"使用COCO预训练权重: {model_name}.pt")
                model = YOLO(f"{model_name}.pt")

            # 获取训练参数
            train_config = self.config['training']
            aug_config = self.config['augmentation']
            output_config = self.config['output']

            # 开始训练
            logger.info("\n开始训练...")
            logger.info(f"Epochs: {train_config['epochs']}")
            logger.info(f"Batch Size: {train_config['batch']}")
            logger.info(f"Image Size: {train_config['imgsz']}")
            logger.info(f"Device: {train_config['device']}")
            logger.info(f"Optimizer: {train_config.get('optimizer', 'auto')}")

            results = model.train(
                # 数据配置
                data=self.config['data']['yaml_path'],

                # 基础训练参数
                epochs=train_config['epochs'],
                imgsz=train_config['imgsz'],
                batch=train_config['batch'],
                device=train_config['device'],
                workers=train_config.get('workers', 8),

                # 项目设置
                project=output_config['project'],
                name=output_config['name'],
                exist_ok=True,

                # 优化器配置
                optimizer=train_config.get('optimizer', 'auto'),
                lr0=train_config.get('lr0', 0.01),
                lrf=train_config.get('lrf', 0.01),
                momentum=train_config.get('momentum', 0.937),
                weight_decay=train_config.get('weight_decay', 0.0005),
                warmup_epochs=train_config.get('warmup_epochs', 3),
                warmup_momentum=train_config.get('warmup_momentum', 0.8),
                warmup_bias_lr=train_config.get('warmup_bias_lr', 0.1),

                # 数据增强
                hsv_h=aug_config.get('hsv_h', 0.015),
                hsv_s=aug_config.get('hsv_s', 0.7),
                hsv_v=aug_config.get('hsv_v', 0.4),
                degrees=aug_config.get('degrees', 0.0),
                translate=aug_config.get('translate', 0.1),
                scale=aug_config.get('scale', 0.5),
                shear=aug_config.get('shear', 0.0),
                perspective=aug_config.get('perspective', 0.0),
                flipud=aug_config.get('flipud', 0.0),
                fliplr=aug_config.get('fliplr', 0.5),
                mosaic=aug_config.get('mosaic', 1.0),
                mixup=aug_config.get('mixup', 0.0),
                copy_paste=aug_config.get('copy_paste', 0.0),

                # 训练控制
                patience=train_config.get('patience', 50),
                save=True,
                save_period=train_config.get('save_period', -1),
                val=True,

                # 可视化
                plots=True,

                # 性能优化
                amp=train_config.get('amp', True),
                cache=train_config.get('cache', False),
                close_mosaic=train_config.get('close_mosaic', 10),

                # 其他
                verbose=True,
                seed=train_config.get('seed', 0),
                deterministic=train_config.get('deterministic', True),
            )

            # 记录结束时间
            end_time = datetime.now()
            duration = end_time - start_time

            logger.info("\n" + "=" * 60)
            logger.info("训练完成！")
            logger.info("=" * 60)
            logger.info(f"训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"总耗时: {duration}")
            logger.info(f"最佳模型: {results.save_dir / 'weights' / 'best.pt'}")
            logger.info(f"最终模型: {results.save_dir / 'weights' / 'last.pt'}")

            # 提取并显示关键指标
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                logger.info("\n关键指标:")
                logger.info(f"  mAP@0.5:      {metrics.get('metrics/mAP50(B)', 0):.4f}")
                logger.info(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
                logger.info(f"  Precision:    {metrics.get('metrics/precision(B)', 0):.4f}")
                logger.info(f"  Recall:       {metrics.get('metrics/recall(B)', 0):.4f}")

            logger.info("\n" + "=" * 60)
            logger.info("下一步:")
            logger.info("=" * 60)
            logger.info(f"1. 查看训练曲线:")
            logger.info(f"   tensorboard --logdir={output_config['project']}")
            logger.info(f"\n2. 评估模型:")
            logger.info(f"   python evaluation/eval_yolo11.py --model {results.save_dir / 'weights' / 'best.pt'}")
            logger.info(f"\n3. 推理测试:")
            logger.info(
                f"   python inference/predict_yolo11.py --model {results.save_dir / 'weights' / 'best.pt'} --source test.jpg")
            logger.info("=" * 60)

            return results

        except KeyboardInterrupt:
            logger.warning("\n训练被用户中断！")
            sys.exit(0)
        except Exception as e:
            logger.error(f"\n训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def print_config(self):
        """打印完整配置"""
        logger.info("\n完整配置:")
        logger.info(yaml.dump(self.config, allow_unicode=True, default_flow_style=False))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 Training Script for RoboMaster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基础训练
    python train_yolo11.py --config .../yolo11_s.yaml

    # 从检查点恢复
    python train_yolo11.py --config configs/yolo11_s.yaml --resume .../last.pt

    # 打印配置后退出
    python train_yolo11.py --config .../yolo11_s.yaml --print-config

YOLOv11 模型选择:
    yolo11n - Nano:   2.6M参数, 超快速
    yolo11s - Small:  9.4M参数, 速度精度平衡 (推荐)
    yolo11m - Medium: 20.1M参数, 更高精度
    yolo11l - Large:  25.3M参数, 最高精度
    yolo11x - XLarge: 56.9M参数, 极致精度
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='训练配置文件路径 (YAML格式)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='恢复训练的检查点路径'
    )

    parser.add_argument(
        '--print-config',
        action='store_true',
        help='打印配置后退出（不训练）'
    )

    return parser.parse_args()


def main():
    """主函数"""

    # 解析参数
    args = parse_args()

    # 创建训练器
    trainer = YOLO11Trainer(args.config)

    # 如果只是打印配置
    if args.print_config:
        trainer.print_config()
        return

    # 如果需要恢复训练
    if args.resume:
        logger.info(f"从检查点恢复训练: {args.resume}")
        trainer.config['model']['pretrained_weights'] = args.resume

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()