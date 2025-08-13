from pathlib import Path
import sys
import time

# Ensure project root in path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline.texts import TextBlurrer
import cv2


def main():
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "jiepai2.jpeg"
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "text_blurred_dbnet.jpg"

    dbnet_root = project_root.parent / "DBNet"
    weights = dbnet_root / "weights" / "best.pt"

    if not weights.exists():
        print(f"缺少 DBNet 权重: {weights}")
        return
    if not input_path.exists():
        print(f"输入图像不存在: {input_path}")
        return

    blurrer = TextBlurrer(
        warmup=True,
        dbnet_root=str(dbnet_root),
        weights_path=str(weights),
        input_size=1280,
        prob_threshold=0.15,
        box_threshold=0.40,
        max_candidates=500,
        min_size=2,
        dilate_px=18,
    )

    # Load original for comparison
    orig = cv2.imread(str(input_path))

    t0 = time.perf_counter()
    result = blurrer.process_image(str(input_path), str(output_path), method="gaussian", max_size=1400)
    t1 = time.perf_counter()
    print(f"Saved to: {result}")
    print(f"First run elapsed: {t1 - t0:.2f}s")

    # second run
    t2 = time.perf_counter()
    _ = blurrer.process_image(str(input_path), str(output_path), method="gaussian", max_size=1400)
    t3 = time.perf_counter()
    print(f"Second run elapsed: {t3 - t2:.2f}s")

    # Save side-by-side comparison
    blurred = cv2.imread(str(output_path))
    if orig is not None and blurred is not None:
        h = 600
        def resize_keep(img, target_h):
            h0, w0 = img.shape[:2]
            w = int(w0 * (target_h / h0))
            return cv2.resize(img, (w, target_h))
        left = resize_keep(orig, h)
        right = resize_keep(blurred, h)
        comparison = cv2.hconcat([left, right])
        comp_path = output_dir / "comparison_text_blurred_dbnet.jpg"
        cv2.imwrite(str(comp_path), comparison)
        print(f"Comparison saved to: {comp_path}")


if __name__ == "__main__":
    main()
