数据脱敏系统（人脸/车牌/街牌）

本项目实现了在 CPU 上的人脸、车牌、街牌文本的检测与“毛玻璃”匿名化（可选像素化/纯色覆盖），并规划了完整的前后端流程。

====================

# 一、功能概述

- 支持的敏感信息：
  - 人脸（RetinaFace 或 YOLOv8-face）
  - 车牌（YOLOv8 车牌模型 + Haar 级联回退）
  - 街牌/文本（本地 DBNet；多边形掩码，支持斜体/旋转文本）
- 脱敏方式：
  - 高斯模糊（毛玻璃，默认，与人脸逻辑一致，按区域自适应核大小）
  - 像素化
  - 纯色覆盖（严格匿名化，用于审计/安全更严格场景）
- CPU 友好：
  - 禁用 GPU，阈值和输入分辨率可调
  - 统一参数与流程，易于集成

目录关键位置：
- 代码：`src/pipeline/`
  - `faces.py`、`plates.py`、`texts.py`
- 测试：`tests/`
  - `test_plate_blur.py`、`test_text_blur.py`
- 数据：`data/`（示例图片）
- 模型：`models/`（人脸/车牌）与 `../DBNet/weights/`（DBNet）
- 输出：`output/`

====================

# 二、环境与依赖

建议使用 Conda 虚拟环境。

```bash
conda create -n data-sensitive-cpu python=3.10 -y
conda activate data-sensitive-cpu

# 依赖（示例）
pip install -U pip
pip install opencv-python numpy torch ultralytics timm shapely pyclipper
pip install retinaface  # 如使用 Keras RetinaFace 备选
```

DBNet 本地仓库：
- 路径：`/home/hui/ws/luoqi/code/data_sensitive/DBNet`
- 权重：放在 `DBNet/weights/best.pt`

====================

# 三、前后端总体流程（产品视角）

1) 用户在前端阅读并勾选《隐私声明与处理协议》。
2) 用户选择是否开启“本地脱敏”：
   - 开启：后端对上传数据先脱敏，再上传至所选服务器。
   - 关闭：后端直接上传原始数据（谨慎）。
3) 用户选择目标服务器区域（如 EU/US 等），影响上传目标地址或桶。
4) 用户选择上传的数据文件夹（前端分片 + 后端流式接收）。
5) 后端处理：
   - 逐张图片执行：人脸/车牌/文本检测 → 多边形/框区域 → 毛玻璃匿名化。
   - 可配置阈值、输入分辨率、膨胀像素、padding 等。
6) 前端展示：
   - 脱敏前后对比图（并排/滑块对比）。
   - 进度与日志。
7) 用户确认：
   - 选择下载脱敏后的数据包，或继续上传到目标服务器。

====================

# 四、后端 API 设计（建议 FastAPI）

说明：当前仓库已具备核心算法与测试脚本。以下为推荐的后端接口设计，便于前端对接。若需要，我可按此规范快速补充后端代码。

- `POST /api/agreements/accept`
  - Body: `{ user_id, accepted: true }`
  - 用途：记录用户协议勾选。

- `POST /api/process/options`
  - Body: `{ anonymize: true, target_region: "EU" }`
  - 返回：会话/任务 ID，用于后续上传与处理。

- `POST /api/upload/{task_id}`
  - 多文件上传（分片可选）。后端落盘到 `uploads/{task_id}/`。

- `POST /api/process/{task_id}`
  - Body（可选覆盖默认参数）：
    ```json
    {
      "max_size": 1400,
      "face": {"backend": "yolov8_face", "method": "gaussian"},
      "plate": {"method": "gaussian"},
      "text": {
        "method": "gaussian",
        "input_size": 1280,
        "prob_threshold": 0.12,
        "dilate_px": 10,
        "pad_ratio": 0.0,
        "use_padded_rect": false
      }
    }
    ```
  - 行为：对 `uploads/{task_id}/` 下文件进行脱敏，输出至 `outputs/{task_id}/`。

- `GET /api/result/{task_id}/preview`
  - 返回：缩略图/对比图列表（分页）。

- `GET /api/result/{task_id}/download`
  - 返回：打包的脱敏结果 zip。

- `POST /api/upload/{task_id}/forward`
  - 将处理后的数据上传到所选区域的云端目标（S3/OSS 等）。

====================

# 五、前端交互要点

- __隐私协议页__：强制勾选才可继续。
- __处理选项页__：
  - 区域选择（EU/US/...）
  - 是否本地脱敏（默认开启）
  - 脱敏参数（可折叠高级设置）
- __上传页__：
  - 文件夹选择，进度显示。
- __预览页__：
  - 并排或滑块对比（本项目测试已生成并排图，参考 `tests/test_text_blur.py`）。
  - 允许单张重处理（例如更强模糊）。
- __导出/上传页__：选择下载脱敏包或上传至指定区域服务器。

====================

# 六、当前可用的本地测试

车牌脱敏：
```bash
conda activate data-sensitive-cpu
PYTHONPATH=. python tests/test_plate_blur.py
```

街牌文本脱敏（DBNet）：
```bash
conda activate data-sensitive-cpu
PYTHONPATH=. python tests/test_text_blur.py
```

输出：
- 脱敏图：`output/text_blurred_dbnet.jpg`
- 对比图：`output/comparison_text_blurred_dbnet.jpg`

提示：
- 如召回不足，可调 `input_size`（更大更清晰）、降低 `prob_threshold`。
- 如覆盖过大，可降 `dilate_px`，并保持 `use_padded_rect=false`（多边形掩码，适合斜体）。

====================

# 七、参数建议与默认值（文本/街牌）

- 检测：`input_size=1280`，`prob_threshold≈0.10~0.15`
- 覆盖：`dilate_px≈8~16`，`pad_ratio=0.0`，`use_padded_rect=false`
- 模糊方式：`method="gaussian"`（“毛玻璃”，与人脸一致）

====================

# 八、后续规划（可选增强）

- 多尺度推理并集，提升小字召回。
- 端到端 FastAPI 实现及 CLI 工具：
  - `cli process --input folder --output folder --text.method gaussian ...`
- 前端滑块对比、任务进度与日志。
- 云端直传/分片/断点续传集成。

====================

# 九、合规与审计

- 严格的匿名化可选 `method="solid"`。
- 记录处理参数与日志，以便审计追踪（任务 ID、时间、参数快照）。

====================

# 十、FAQ

- 斜体文本遮挡过大？
  - 关闭 `use_padded_rect`（默认即关闭），仅用多边形掩码；并调低 `dilate_px`。
- 文字仍可辨识？
  - 改用 `method="solid"` 或加大模糊核（我可将核大小暴露为参数）。