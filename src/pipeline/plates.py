import cv2
import numpy as np
import os
from typing import List, Tuple, Optional


class PlateBlurrer:
    def __init__(self, warmup: bool = True, backend: str = "yolov8_plate", weights_path: Optional[str] = None,
                 cascade_path: Optional[str] = None):
        """
        车牌检测与模糊（与 FaceBlurrer 流程一致）。

        参数:
          warmup: 是否在初始化时进行一次小尺寸预热
          backend: 检测后端，支持 "yolov8_plate" | "haarcascade_plate"
          weights_path: YOLOv8 车牌检测权重路径（默认 models/license_plate_detector.pt）
          cascade_path: Haarcascade XML 路径（默认 models/haarcascade_russian_plate_number.xml）
        """
        # CPU 运行与静默日志
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

        self.backend = backend
        self.weights_path = weights_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                                         "models", "license_plate_detector.pt")
        self.cascade_path = cascade_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                                         "models", "haarcascade_russian_plate_number.xml")
        self.detector = self._load_detector()

        if warmup:
            try:
                dummy = np.zeros((160, 320, 3), dtype=np.uint8)
                _ = self.detect_plates(dummy)
            except Exception:
                pass

    def _load_detector(self):
        if self.backend == "yolov8_plate":
            try:
                from ultralytics import YOLO
            except Exception as e:
                raise ImportError("需要安装 ultralytics: pip install ultralytics") from e
            if not (self.weights_path and os.path.exists(self.weights_path)):
                # 回退到 Haarcascade
                if os.path.exists(self.cascade_path):
                    self.backend = "haarcascade_plate"
                    return cv2.CascadeClassifier(self.cascade_path)
                raise FileNotFoundError("未找到 YOLOv8 车牌权重且无可用 Haarcascade。")
            return YOLO(self.weights_path)

        if self.backend == "haarcascade_plate":
            if not os.path.exists(self.cascade_path):
                raise FileNotFoundError(f"未找到级联文件: {self.cascade_path}")
            return cv2.CascadeClassifier(self.cascade_path)

        raise ValueError(f"不支持的后端: {self.backend}")

    def detect_plates(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        boxes: List[Tuple[int, int, int, int]] = []
        if self.backend == "yolov8_plate":
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector(img_rgb, conf=0.25, iou=0.5, verbose=False)
            if results and len(results) > 0:
                r0 = results[0]
                if hasattr(r0, "boxes") and r0.boxes is not None:
                    arr = r0.boxes.xyxy.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2) in arr:
                        boxes.append((int(x1), int(y1), int(x2), int(y2)))
        elif self.backend == "haarcascade_plate":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 10))
            for (x, y, w, h) in rects:
                boxes.append((int(x), int(y), int(x + w), int(y + h)))
        return boxes

    def blur_plates(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]], method: str = "gaussian") -> np.ndarray:
        for (x1, y1, x2, y2) in boxes:
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = image[y1:y2, x1:x2]

            if method == "pixelate":
                pixel_size = max(1, min(roi.shape[0] // 10, roi.shape[1] // 10))
                small = cv2.resize(roi, (max(1, roi.shape[1] // pixel_size), max(1, roi.shape[0] // pixel_size)),
                                   interpolation=cv2.INTER_NEAREST)
                blurred = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                ksize = max(3, min(roi.shape[0] // 4, roi.shape[1] // 4) * 2 + 1)
                blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)

            image[y1:y2, x1:x2] = blurred
        return image

    def process_image(self, input_path: str, output_path: str, method: str = "gaussian", max_size: int = 640) -> str:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"无法读取图像: {input_path}")
        orig_image = image.copy()
        orig_h, orig_w = image.shape[:2]
        scale = min(max_size / orig_w, max_size / orig_h, 1.0)
        if scale < 1:
            image = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)))

        boxes = self.detect_plates(image)
        print(f"检测到 {len(boxes)} 个车牌")
        if boxes:
            image = self.blur_plates(image, boxes, method)

        if scale < 1:
            image = cv2.resize(image, (orig_w, orig_h))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ok = cv2.imwrite(output_path, image)
        if not ok:
            raise IOError(f"无法写入输出文件: {output_path}")
        print(f"处理完成，结果保存至: {output_path}")

        self.show_results(orig_image, image)
        return output_path

    def show_results(self, orig_image: np.ndarray, processed_image: np.ndarray):
        try:
            display_size = (800, 600)
            orig_display = cv2.resize(orig_image, display_size)
            processed_display = cv2.resize(processed_image, display_size)
            comparison = np.hstack((orig_display, processed_display))
            cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "Blurred", (display_size[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Plate Blurring Results", comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass
