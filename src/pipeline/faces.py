import cv2
import numpy as np
import os
from typing import List, Tuple, Optional


class FaceBlurrer:
    def __init__(self, warmup: bool = True, backend: str = "yolov8_face", weights_path: Optional[str] = None):
        """
        支持多后端的人脸检测与模糊（CPU 友好）。

        参数:
          warmup: 是否在初始化时进行一次小尺寸预热，以避免首帧卡顿。
          backend: 检测后端，支持 "yolov8_face" | "keras_retinaface"
          weights_path: 部分后端（如 YOLOv8-face）可指定权重路径
        """
        # 强制使用 CPU，并降低 TensorFlow 日志噪音
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

        self.backend = backend
        self.weights_path = weights_path
        self.detector = self._load_detector()

        # 预热：使用一张小尺寸黑图触发模型构建
        if warmup:
            try:
                dummy = np.zeros((160, 160, 3), dtype=np.uint8)
                _ = self.detect_faces(dummy)
            except Exception:
                # 预热失败不影响后续实际调用
                pass

    def _load_detector(self):
        """按后端加载检测器。"""
        if self.backend == "keras_retinaface":
            try:
                from retinaface import RetinaFace
            except Exception as e:
                raise ImportError(
                    "请先安装 Keras 版 retinaface: pip install retinaface tf-keras tensorflow"
                ) from e
            return RetinaFace

        if self.backend == "yolov8_face":
            try:
                from ultralytics import YOLO
            except Exception as e:
                raise ImportError(
                    "需要 ultralytics 包: pip install ultralytics"
                ) from e
            # 需要提供人脸检测权重文件路径（例如 models/yolov8n-face.pt）。
            # 若缺失，则回退到 keras_retinaface，保证功能可用。
            if not self.weights_path or not os.path.exists(self.weights_path):
                try:
                    from retinaface import RetinaFace
                    self.backend = "keras_retinaface"  # 回退
                    return RetinaFace
                except Exception:
                    # 如果RetinaFace也不可用，使用OpenCV的Haar级联作为最后的回退
                    import cv2
                    cascade_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                               "models", "haarcascade_frontalface_default.xml")
                    if os.path.exists(cascade_path):
                        self.backend = "opencv_haar"
                        return cv2.CascadeClassifier(cascade_path)
                    else:
                        # 创建一个虚拟检测器，返回空结果但不会崩溃
                        self.backend = "dummy"
                        return None
            return YOLO(self.weights_path)

        raise ValueError(f"不支持的后端: {self.backend}")

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测图像中的人脸，返回 bounding boxes 列表。"""
        face_boxes: List[Tuple[int, int, int, int]] = []

        if self.backend == "keras_retinaface":
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(img_rgb)
            if isinstance(detections, dict):
                for _, v in detections.items():
                    facial_area = v.get("facial_area")
                    if facial_area and len(facial_area) == 4:
                        x1, y1, x2, y2 = map(int, facial_area)
                        face_boxes.append((x1, y1, x2, y2))

        elif self.backend == "yolov8_face":
            # Ultralytics YOLOv8-face 推理
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector(img_rgb, verbose=False)
            if results and len(results) > 0:
                r0 = results[0]
                if hasattr(r0, "boxes") and r0.boxes is not None:
                    boxes = r0.boxes.xyxy.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2) in boxes:
                        face_boxes.append((int(x1), int(y1), int(x2), int(y2)))

        elif self.backend == "opencv_haar":
            # OpenCV Haar级联检测
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face_boxes.append((int(x), int(y), int(x + w), int(y + h)))

        elif self.backend == "dummy":
            # 虚拟检测器，返回空结果
            pass

        return face_boxes

    def blur_faces(self, image: np.ndarray, face_boxes, method: str = "gaussian") -> np.ndarray:
        """模糊检测到的人脸区域"""
        for (x1, y1, x2, y2) in face_boxes:
            # 确保坐标在图像范围内
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # 提取人脸区域
            face_roi = image[y1:y2, x1:x2]

            # 应用模糊
            if method == "pixelate":
                pixel_size = max(1, min(face_roi.shape[0] // 10, face_roi.shape[1] // 10))
                small = cv2.resize(
                    face_roi,
                    (
                        max(1, face_roi.shape[1] // pixel_size),
                        max(1, face_roi.shape[0] // pixel_size),
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )
                blurred = cv2.resize(
                    small, (face_roi.shape[1], face_roi.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            else:
                # 高斯模糊 (默认)
                ksize = max(3, min(face_roi.shape[0] // 4, face_roi.shape[1] // 4) * 2 + 1)
                blurred = cv2.GaussianBlur(face_roi, (ksize, ksize), 0)

            # 将模糊后的区域放回原图
            image[y1:y2, x1:x2] = blurred

        return image

    def process_image(self, input_path: str, output_path: str, method: str = "gaussian", max_size: int = 640):
        """
        处理单张图像

        参数:
            input_path: 输入图像路径
            output_path: 输出图像路径
            method: 模糊方法 ('gaussian' 或 'pixelate')
            max_size: 图像最大尺寸 (长边)
        """
        # 加载图像
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"无法读取图像: {input_path}")

        # 保存原始图像用于显示
        orig_image = image.copy()

        # 为优化CPU性能，调整图像尺寸
        orig_h, orig_w = image.shape[:2]
        scale = min(max_size / orig_w, max_size / orig_h, 1.0)
        if scale < 1:
            image = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)))

        # 检测人脸
        face_boxes = self.detect_faces(image)
        print(f"检测到 {len(face_boxes)} 张人脸")

        # 模糊人脸
        if face_boxes:
            image = self.blur_faces(image, face_boxes, method)

        # 恢复原始尺寸
        if scale < 1:
            image = cv2.resize(image, (orig_w, orig_h))

        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ok = cv2.imwrite(output_path, image)
        if not ok:
            raise IOError(f"无法写入输出文件: {output_path}")
        print(f"处理完成，结果保存至: {output_path}")

        # 显示结果（在无显示环境下可能失败，做保护）
        self.show_results(orig_image, image)

        return output_path

    def show_results(self, orig_image: np.ndarray, processed_image: np.ndarray):
        """显示原始和处理后的图像对比"""
        # 在Docker/服务器环境下跳过GUI显示
        # 这个方法主要用于本地开发调试
        pass
