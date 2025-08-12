from __future__ import annotations
import time
import os
from pathlib import Path

from typing import List, Tuple

# Ensure project root on sys.path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pipeline.faces import FaceBlurrer


def run_once(backend: str, img_path: Path, out_path: Path, weights_path: str | None = None) -> float:
    t0 = time.perf_counter()
    blurrer = FaceBlurrer(warmup=True, backend=backend, weights_path=weights_path)
    t1 = time.perf_counter()

    # first run
    t2 = time.perf_counter()
    blurrer.process_image(str(img_path), str(out_path), method="gaussian", max_size=640)
    t3 = time.perf_counter()

    # second run
    t4 = time.perf_counter()
    blurrer.process_image(str(img_path), str(out_path), method="gaussian", max_size=640)
    t5 = time.perf_counter()

    init = t1 - t0
    first = t3 - t2
    second = t5 - t4
    return init, first, second


def main():
    img = ROOT / "data" / "person.jpeg"
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    backends = [
        ("keras_retinaface", None),
        ("yolov8_face", str(ROOT / "models" / "yolov8n-face.pt")),
        ("pytorch_retinaface", str(ROOT / "models" / "retinaface_resnet50.pth")),
    ]

    results = []
    for backend, weights in backends:
        print(f"\n==== Benchmark {backend} ====")
        try:
            init, first, second = run_once(
                backend,
                img_path=img,
                out_path=out_dir / f"bench_{backend}.jpg",
                weights_path=weights,
            )
            print(f"Init:   {init:.2f}s  First: {first:.2f}s  Second: {second:.2f}s")
            results.append((backend, init, first, second))
        except FileNotFoundError as e:
            print(f"SKIP {backend}: {e}")
        except ImportError as e:
            print(f"SKIP {backend}: {e}")
        except NotImplementedError as e:
            print(f"SKIP {backend}: {e}")
        except Exception as e:
            print(f"ERROR {backend}: {e}")

    print("\n==== Summary ====")
    for b, init, f, s in results:
        print(f"{b:20s} | init {init:.2f}s | first {f:.2f}s | second {s:.2f}s")


if __name__ == "__main__":
    main()
