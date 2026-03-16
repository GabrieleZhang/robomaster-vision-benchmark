import time
import numpy as np
import pandas as pd
from pathlib import Path
import openvino as ov

# =========================
# 配置
# =========================

MODELS = {
    "yolo11s": r".../best_openvino_model",
    "yolo26s": r".../best_openvino_model",
    "rtdetr_l": r".../best_openvino_model",
}

IMGSZ = 640
BATCH = 1

WARMUP = 50
ITERS = 300

THREADS = 8


# =========================
# 工具函数
# =========================

def load_openvino_model(model_dir):

    model_dir = Path(model_dir)

    xml_path = model_dir / "best.xml"
    bin_path = model_dir / "best.bin"

    if not xml_path.exists():
        raise RuntimeError(f"model.xml not found: {xml_path}")

    if not bin_path.exists():
        raise RuntimeError(f"model.bin not found: {bin_path}")

    core = ov.Core()

    model = core.read_model(xml_path)

    compiled_model = core.compile_model(
        model,
        "CPU",
        {
            "PERFORMANCE_HINT": "THROUGHPUT",
            "NUM_STREAMS": "AUTO"
        }
    )

    return compiled_model


def create_dummy_input(compiled_model):

    input_layer = compiled_model.input(0)
    shape = input_layer.shape

    dummy = np.random.rand(*shape).astype(np.float32)

    return dummy


def run_benchmark(compiled_model, dummy):

    # warmup
    for _ in range(WARMUP):
        compiled_model([dummy])

    times = []

    for _ in range(ITERS):

        start = time.perf_counter()

        compiled_model([dummy])

        end = time.perf_counter()

        times.append((end - start) * 1000)

    times = np.array(times)

    latency_mean = times.mean()
    latency_std = times.std()

    fps = 1000 / latency_mean

    return {
        "Latency_mean_ms": latency_mean,
        "Latency_std_ms": latency_std,
        "FPS": fps,
    }


# =========================
# 主函数
# =========================

def main():

    print("=" * 60)
    print("OpenVINO CPU Benchmark")
    print("=" * 60)

    results = []

    for name, model_path in MODELS.items():

        print(f"\n===== {name} =====")

        try:

            compiled_model = load_openvino_model(model_path)

            dummy = create_dummy_input(compiled_model)

            perf = run_benchmark(compiled_model, dummy)

            print(f"Latency: {perf['Latency_mean_ms']:.2f} ms")
            print(f"FPS: {perf['FPS']:.2f}")

            results.append({
                "Model": name,
                **perf
            })

        except Exception as e:

            print("FAILED:", e)

    if len(results) == 0:
        print("All tests failed")
        return

    df = pd.DataFrame(results)

    print("\n================ RESULT ================")
    print(df.to_string(index=False))
    print("========================================")

    df.to_csv("cpu_benchmark_results.csv", index=False)

    print("\nResults saved to cpu_benchmark_results.csv")


if __name__ == "__main__":
    main()