from pathlib import Path
import sys
import time

# Add project root for module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline.plates import PlateBlurrer


def main():
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "carpai.jpeg"
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "plate_blurred_yolov8.jpg"

    # Prefer YOLOv8 weights from the ALPR repo; fallback to Haarcascade
    yolov8_weights = project_root / "models" / "license_plate_detector.pt"
    cascade_xml = project_root / "models" / "haarcascade_russian_plate_number.xml"

    if yolov8_weights.exists():
        blurrer = PlateBlurrer(
            warmup=True,
            backend="yolov8_plate",
            weights_path=str(yolov8_weights),
        )
        print("Using backend: yolov8_plate")
    elif cascade_xml.exists():
        blurrer = PlateBlurrer(
            warmup=True,
            backend="haarcascade_plate",
            cascade_path=str(cascade_xml),
        )
        print("Using backend: haarcascade_plate")
    else:
        print("缺少权重: 请放置 models/license_plate_detector.pt 或 models/haarcascade_russian_plate_number.xml")
        return

    if not input_path.exists():
        print(f"输入图像不存在: {input_path}")
        print("请将测试图片放至 data/car.jpg 或修改此脚本中的路径。")
        return

    t0 = time.perf_counter()
    result = blurrer.process_image(str(input_path), str(output_path), method="gaussian", max_size=640)
    t1 = time.perf_counter()
    print(f"Saved to: {result}")
    print(f"First run elapsed: {t1 - t0:.2f}s")

    # Second run to show warmup/model build effect
    t2 = time.perf_counter()
    _ = blurrer.process_image(str(input_path), str(output_path), method="gaussian", max_size=640)
    t3 = time.perf_counter()
    print(f"Second run elapsed: {t3 - t2:.2f}s")


if __name__ == "__main__":
    main()
