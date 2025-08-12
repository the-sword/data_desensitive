from pathlib import Path

try:
    from src.pipeline.faces import FaceBlurrer
except ModuleNotFoundError:
    # Allow running without PYTHONPATH by adding project root
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.pipeline.faces import FaceBlurrer


def main():
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "person.jpeg"
    output_dir = project_root / "output"
    output_path = output_dir / "face_blurred_yolov8.jpg"

    import time

    blurrer = FaceBlurrer(
        warmup=True,
        backend="yolov8_face",
        weights_path=str(project_root / "models" / "yolov8n-face.pt"),
    )

    t0 = time.perf_counter()
    result = blurrer.process_image(str(input_path), str(output_path), method="gaussian")
    t1 = time.perf_counter()
    print(f"Saved to: {result}")
    print(f"First run elapsed: {t1 - t0:.2f}s")

    # Second run to show speed-up after warmup/model build
    t2 = time.perf_counter()
    result2 = blurrer.process_image(str(input_path), str(output_path), method="gaussian")
    t3 = time.perf_counter()
    print(f"Second run elapsed: {t3 - t2:.2f}s")


if __name__ == "__main__":
    main()
