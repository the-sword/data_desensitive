import cv2
import numpy as np
import os
from typing import List, Tuple


class FaceBlurrer:
    def __init__(self, warmup: bool = True, backend: str = "keras_retinaface", weights_path: str | None = None):
        """
        使用 Keras 版 RetinaFace 进行人脸检测与模糊（CPU 友好）。

        参数:
          warmup: 是否在初始化时进行一次小尺寸预热，以避免首帧卡顿。
          backend: 检测后端，支持 "keras_retinaface" | "yolov8_face" | "pytorch_retinaface"
          weights_path: 部分后端（如 PyTorch RetinaFace）可指定权重路径
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
            # 需要提供人脸检测权重文件路径（例如 models/yolov8n-face.pt）
            if not self.weights_path or not os.path.exists(self.weights_path):
                raise FileNotFoundError(
                    "未提供 YOLOv8-face 权重文件。请将人脸模型权重保存到本地并通过 weights_path 传入，例如 models/yolov8n-face.pt"
                )
            return YOLO(self.weights_path)

        if self.backend == "pytorch_retinaface":
            # 采用 biubug6/Pytorch_Retinaface 实现；需要其源码可被导入
            try:
                import torch  # noqa: F401
                # 延迟导入并在 detect_faces 中具体使用，以减少此处依赖复杂度
                return "pytorch_retinaface"
            except Exception as e:
                raise ImportError(
                    "需要 PyTorch 以及 biubug6/Pytorch_Retinaface 实现。建议安装：\n"
                    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
                    "pip install git+https://github.com/biubug6/Pytorch_Retinaface.git"
                ) from e

        raise ValueError(f"不支持的后端: {self.backend}")

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测图像中的人脸，返回 bounding boxes 列表。"""
        # Keras 版 RetinaFace 接口：RetinaFace.detect_faces(img) -> dict
        # 每个项包含 'facial_area': [x1, y1, w, h]        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

        elif self.backend == "pytorch_retinaface":
            # 这里给出最小可用实现框架；具体加载模型定义与权重可能因仓库版本而异
            try:
                import torch
                from retinaface.data import cfg_re50
                from retinaface.models.retinaface import RetinaFace as RFModel
                from retinaface.utils.prior_box import PriorBox
                from retinaface.utils.nms.py_cpu_nms import py_cpu_nms
                from retinaface.layers.functions.prior_box import PriorBox as PriorBoxLegacy  # 兼容不同分支
            except Exception:
                raise ImportError(
                    "未找到 biubug6/Pytorch_Retinaface 相关模块。请安装后重试。"
                )

            if not self.weights_path or not os.path.exists(self.weights_path):
                raise FileNotFoundError(
                    "未提供有效的 PyTorch RetinaFace 权重路径 weights_path，例如 models/retinaface_resnet50.pth"
                )

            # 为避免频繁加载，这里简单地每次创建模型；生产环境应缓存到 self.detector
            device = torch.device("cpu")
            cfg = cfg_re50
            net = RFModel(cfg=cfg, phase="test")
            state_dict = torch.load(self.weights_path, map_location=device)
            net.load_state_dict(state_dict, strict=False)
            net.eval().to(device)

            # 推理前处理，具体实现依赖仓库；此处省略详细预处理、解码与 NMS 实现
            # 为保持任务推进，这里暂时抛出提示，指引安装后以专用脚本进行对比。
            raise NotImplementedError(
                "为了保证兼容你本地的权重与仓库实现，建议安装 biubug6 代码后，我将补充完备的预处理/后处理并缓存模型。"
            )

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
