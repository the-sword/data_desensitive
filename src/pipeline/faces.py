import cv2
import numpy as np
import os
from typing import List, Tuple


class FaceBlurrer:
    def __init__(self):
        """使用 Keras 版 RetinaFace 进行人脸检测与模糊（CPU 友好）。"""
        self.detector = self._load_detector()

    def _load_detector(self):
        """加载 Keras 版 RetinaFace（无需本地权重）。"""
        try:
            from retinaface import RetinaFace  # type: ignore
        except ImportError:
            raise ImportError(
                "未找到 Keras 版 RetinaFace 包。请安装: pip install retinaface tf-keras"
            )
        # 返回模块以便后续直接调用静态方法
        return RetinaFace

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测图像中的人脸，返回 bounding boxes 列表。"""
        # Keras 版 RetinaFace 接口：RetinaFace.detect_faces(img) -> dict
        # 每个项包含 'facial_area': [x1, y1, w, h] 或 [x1, y1, x2, y2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(img_rgb)
        boxes: List[Tuple[int, int, int, int]] = []
        if isinstance(detections, dict):
            for _, det in detections.items():
                area = det.get("facial_area", None)
                if not area:
                    continue
                if len(area) == 4:
                    x1, y1, x2, y2 = area
                    # 某些实现可能返回 (x, y, w, h)
                    if x2 <= x1 or y2 <= y1:
                        x, y, w, h = area
                        x1, y1, x2, y2 = x, y, x + w, y + h
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes

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

    def process_image(self, input_path: str, output_path: str, method: str = "gaussian", max_size: int = 1200):
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
        try:
            display_size = (800, 600)
            orig_display = cv2.resize(orig_image, display_size)
            processed_display = cv2.resize(processed_image, display_size)

            # 创建对比图像
            comparison = np.hstack((orig_display, processed_display))

            # 添加标题
            cv2.putText(
                comparison,
                "Original",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                comparison,
                "Blurred",
                (display_size[0] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Face Blurring Results", comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            # 在无GUI环境下静默跳过
            pass
