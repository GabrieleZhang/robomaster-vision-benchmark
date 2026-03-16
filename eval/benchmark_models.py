import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO, RTDETR
from thop import profile
import matplotlib.pyplot as plt


CONFIG = {
    ".../dataset.yaml",
    "imgsz": 640,
    "device": 0,
    "conf": 0.25,
    "iou": 0.7,
    "warmup": 20,
    "iters": 100,
    "models": {
        "RT-DETR-L": {
            "path": ".../best.pt",
            "type": "rtdetr"
        },
        "YOLO26-S": {
            "path": ".../best.pt",
            "type": "yolo"
        },
        "YOLOv11-S": {
            "path": ".../best.pt",
            "type": "yolo"
        }
    },
    "output": "benchmark_results"
}


class Benchmark:

    def __init__(self, cfg):
        self.cfg = cfg
        self.out = Path(cfg["output"])
        self.out.mkdir(exist_ok=True)

    # ------------------------------
    # 加载模型
    # ------------------------------

    def load_model(self, info):

        if info["type"] == "rtdetr":
            model = RTDETR(info["path"])
        else:
            model = YOLO(info["path"])

        return model

    # ------------------------------
    # 精度评估
    # ------------------------------

    def evaluate_accuracy(self, model):

        results = model.val(
            data=self.cfg["dataset"],
            imgsz=self.cfg["imgsz"],
            batch=1,
            device=self.cfg["device"],
            conf=self.cfg["conf"],
            iou=self.cfg["iou"],
            verbose=False
        )

        m = results.results_dict

        return {
            "mAP50": float(m["metrics/mAP50(B)"]),
            "mAP50-95": float(m["metrics/mAP50-95(B)"]),
            "Precision": float(m["metrics/precision(B)"]),
            "Recall": float(m["metrics/recall(B)"])
        }

    # ------------------------------
    # 真实推理速度
    # ------------------------------

    def evaluate_speed(self, model):

        img = np.random.randint(
            0, 255, (self.cfg["imgsz"], self.cfg["imgsz"], 3), dtype=np.uint8
        )

        device = next(model.model.parameters()).device

        # warmup
        for _ in range(self.cfg["warmup"]):
            model.predict(img, imgsz=self.cfg["imgsz"], verbose=False)

        times = []

        for _ in range(self.cfg["iters"]):

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            model.predict(img, imgsz=self.cfg["imgsz"], verbose=False)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            times.append((end - start) * 1000)

        times = np.array(times)

        latency = times.mean()
        fps = 1000 / latency

        return {
            "Latency_ms": latency,
            "Latency_std": times.std(),
            "FPS": fps
        }

    # ------------------------------
    # 模型复杂度
    # ------------------------------

    def evaluate_complexity(self, model):

        device = next(model.model.parameters()).device

        dummy = torch.randn(1, 3, self.cfg["imgsz"], self.cfg["imgsz"]).to(device)

        flops, params = profile(model.model, inputs=(dummy,), verbose=False)

        return {
            "Params_M": params / 1e6,
            "GFLOPs": flops / 1e9
        }

    # ------------------------------
    # 运行 benchmark
    # ------------------------------

    def run(self):

        records = []

        for name, info in self.cfg["models"].items():

            print("\nEvaluating:", name)

            model = self.load_model(info)

            acc = self.evaluate_accuracy(model)

            spd = self.evaluate_speed(model)

            comp = self.evaluate_complexity(model)

            result = {"Model": name, **acc, **spd, **comp}

            records.append(result)

            del model
            torch.cuda.empty_cache()

        df = pd.DataFrame(records)

        return df

    # ------------------------------
    # 输出报告
    # ------------------------------

    def export(self, df):

        csv_path = self.out / "benchmark.csv"
        df.to_csv(csv_path, index=False)

        print("\nSaved:", csv_path)

        # LaTeX 表
        latex = df.to_latex(index=False, float_format="%.3f")

        with open(self.out / "table.tex", "w") as f:
            f.write(latex)

        # Speed-Accuracy 图
        plt.figure()

        for _, r in df.iterrows():
            plt.scatter(r["FPS"], r["mAP50"], s=r["Params_M"]*20)
            plt.text(r["FPS"], r["mAP50"], r["Model"])

        plt.xlabel("FPS")
        plt.ylabel("mAP@0.5")
        plt.title("Speed-Accuracy Trade-off")

        plt.savefig(self.out / "speed_accuracy.png", dpi=300)


def main():

    bench = Benchmark(CONFIG)

    df = bench.run()

    print("\nBenchmark Result\n")
    print(df)

    bench.export(df)


if __name__ == "__main__":
    main()