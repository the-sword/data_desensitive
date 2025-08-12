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
    output_path = output_dir / "face_blurred_retinaface.jpg"

    blurrer = FaceBlurrer()
    result = blurrer.process_image(str(input_path), str(output_path), method="gaussian")
    print(f"Saved to: {result}")


if __name__ == "__main__":
    main()
