## 项目简介

本仓库实现了一个基于多模态大模型的 2D→3D affordance 分析与 3D 掩码训练 pipeline。  
整体流程为：

1. 使用 Qwen 按文本指令解析出目标物体和对应的 affordance（如 handle、knob）。
2. 使用 Qwen 处理视频流，从中选出与上述物体与 affordance 最相关的 top‑K 帧，并预测 2D 操作点（points）。
3. 使用 Molmo + SAM / SAM2 在这些关键帧上获得操作区域的 2D mask，并将局部 mask 映射回原始大图像。
4. 将 2D mask 投影到 3D 点云：得到 3D 掩码并提取其邻域点云。
5. 基于提取到的 3D 局部点云，在 **不使用 diffusion 模块** 的前提下，训练 3D affordance / 接触区域预测模型。

本 README 主要说明代码结构、依赖安装、基本数据格式和典型运行方式，便于在 GitHub 上复现与扩展。

---

## 代码结构

- `pipeline/`
  - `step1_affordance/qwen.py`  
    使用 Qwen 文本模型解析自然语言指令，抽取目标物体和 affordance 标签，输出结构化 JSON。
  - `step2_clipwithaffordance/test_clip4_affordance.py`  
    使用 CLIP 将文本 affordance 与图像帧特征对齐，从视频帧中选择最相关的 top‑K 图像。
  - `step3_point_prediction/test_qwen_point.py`  
    使用 Qwen‑VL 模型在选出的图像上预测操作点坐标（2D keypoints）。
  - `step4_crop_images/qwen_seg_image.py`  
    根据操作点和目标物体，裁剪局部图像 patch，为后续分割与三维处理做准备。
  - `step5_molmo_sam/test_molmo.py`  
    使用 Molmo + SAM / SAM2 对局部图像做分割，得到 affordance 相关的 2D mask。
  - `step6_molmo_merge/test_molmo_merge.py`  
    合并多帧 / 多 mask 结果，将局部 mask 映射回原始大图像坐标系。
  - `step7_lift_3d/test_molmo_lift_2d_to_3d.py`  
    利用相机位姿和深度信息，将 2D mask 投影到 3D 点云空间，生成 3D mask 并提取邻域点云。
  - `step8_3d_training/train_no_diff.py`  
    主训练脚本：基于 `AffordanceDataset` 和无 diffusion 头，对 3D 掩码和点云进行训练。
  - `step8_3d_training/test_no_diff.py`  
    对训练好的 3D 模型做评估，使用 `Segment3DEvaluator` 输出 3D segmentation / contact 相关指标。

- `dataset/`
  - `AffordanceDataset.py`  
    主数据集定义，将视频帧、图像、点云、2D/3D mask 等组织成可训练的样本；支持多种配置选项（如是否使用 SAM2 结果、不同预处理版本等）。
  - `preprocess_data_sam2.py`  
    针对 SAM2 输出进行数据预处理，将其转换为 `AffordanceDataset` 可直接使用的格式。
  - `misc.py`  
    包含通用的 `collate_fn_general` 等 dataloader 相关工具。
  - `data_parser_paths.py`  
    将数据资产 ID 映射到磁盘路径。

- `models/`
  - `scene_models/pointtransformer.py`, `scene_models/pointops.py`  
    3D 点云特征抽取 backbone（PointTransformerSeg）。
  - `cdm.py`, `modules.py`, `functions.py`  
    CDM 模型结构、注意力模块以及与 CLIP/scene model 相关的加载与封装。
  - `diffusion/`  
    diffusion 模型与采样/损失定义（当前主 3D 训练入口 `train_no_diff.py` **不启用 diffusion**，但代码保留用于扩展）。
  - `base.py`  
    模型与 diffusion 构建入口 `create_model_and_diffusion`，与 Hydra 配置联动。

- `scenefun3d_utils/`
  - `data_parser.py` (+ `data_parser_paths.py`)  
    负责读取场景数据集（点云、深度、相机参数等），并提供统一接口给 2D→3D lifting 步骤。
  - `fusion_util.py`, `pc_process.py`, `homogenous.py`, `rigid_interpolation.py`  
    点云融合、点云处理、坐标变换与刚体插值等几何工具。
  - `viz.py`, `viz_constants.py`  
    使用 Open3D / pyviz3d 对 3D 点云与 mask 进行可视化。

- 其它
  - `utils/`  
    日志与可视化接口（`io.py`）、训练 loop（`training.py`）、评估器（`evaluator.py`）、注册器（`registry.py`）、各类 metrics 等。
  - `outputs/`  
    训练运行产生的实验目录：每个子目录对应一次 Hydra 实验（含 `log/`, `ckpt/`, `eval/` 等）。
  - `scripts/train.sh`  
    示例训练脚本，调用 `train_no_diff.py` 并指定关键 Hydra 参数。
  - `requirements.txt`  
    Python 依赖列表（见下文安装说明）。

---

## 环境安装

建议使用 Python 3.10+ 和虚拟环境（conda 或 venv）。

```bash
git clone https://github.com/<your-username>/affseg.git
cd affseg

# 建议创建虚拟环境
# conda create -n affseg python=3.10
# conda activate affseg

pip install -r requirements.txt
```

> 说明：
> - `pointops_cuda`（PointTransformer 相关）可能需要单独编译，请根据你本机的 CUDA / PyTorch 版本编译安装（参考 PointTransformer 官方说明或本项目后续补充文档）。
> - 某些大型模型（Qwen, Molmo, SAM2 等）依赖的权重不会随仓库分发，需要通过 `transformers` 或官方仓库自动下载。

---

## 数据准备（简要）

1. **原始数据组织**
   - 视频帧 / 图像
   - 深度图与相机参数（用于 2D→3D 投影）
   - 原始点云 / 场景数据（Scenefun3D 或类似）
2. **路径配置**
   - 在 `dataset/data_parser_paths.py` 与 `scenefun3d_utils/data_parser_paths.py` 中配置真实数据根目录和资产 ID 到路径的映射。
3. **SAM2 / mask 预处理（可选）**
   - 使用 `dataset/preprocess_data_sam2.py` 将 SAM2 输出整理为 `AffordanceDataset` 支持的中间格式。

具体目录结构和字段格式可以参考 `AffordanceDataset.py` 与 `preprocess_data_sam2.py` 中的注释与读取逻辑。

---

## 典型使用流程

### 1. 文本指令 → 物体 & affordance

```bash
python pipeline/step1_affordance/qwen.py \
  --input_instructions path/to/instructions.json \
  --output path/to/affordance.json
```

输出包含目标物体类别与 affordance 标签（例：door handle, drawer handle 等）。

### 2. 视频帧筛选 & 操作点预测

- 使用 CLIP 挑选与 affordance 最相关的 top‑K 帧：

```bash
python pipeline/step2_clipwithaffordance/test_clip4_affordance.py \
  --frames_dir path/to/frames \
  --affordance_json path/to/affordance.json \
  --output path/to/topk_frames.json
```

- 使用 Qwen‑VL 在这些帧上预测操作点：

```bash
python pipeline/step3_point_prediction/test_qwen_point.py \
  --frames path/to/topk_frames.json \
  --output path/to/points.json
```

### 3. 2D 分割与 2D mask 合并

```bash
# 使用 Molmo + SAM / SAM2 做分割
python pipeline/step5_molmo_sam/test_molmo.py \
  --frames path/to/topk_frames.json \
  --points path/to/points.json \
  --output path/to/2d_masks

# 合并局部 mask 回原图
python pipeline/step6_molmo_merge/test_molmo_merge.py \
  --masks_dir path/to/2d_masks \
  --output path/to/merged_masks
```

### 4. 2D→3D 提升与点云提取

```bash
python pipeline/step7_lift_3d/test_molmo_lift_2d_to_3d.py \
  --config path/to/lift_config.yaml \
  --masks path/to/merged_masks \
  --output path/to/3d_masks_and_pcd
```

该步骤会调用 `scenefun3d_utils` 内的几何与解析工具，将 2D mask 投影到 3D，并生成局部点云及 3D 掩码。

### 5. 无 diffusion 的 3D 训练与评估

- **训练（关键入口）**

使用 Hydra 启动 `train_no_diff.py`，可以参考项目中的 `scripts/train.sh`：

```bash
bash scripts/train.sh perceiver_division_8192_pointtransformer_with_new_loss
```

或直接调用：

```bash
CUDA_VISIBLE_DEVICES=0 python pipeline/step8_3d_training/train_no_diff.py \
  exp_name=perceiver_division_8192_pointtransformer_with_new_loss \
  output_dir=outputs \
  platform=TensorBoard \
  diffusion.steps=500 \
  task=contact_gen \
  task.train.batch_size=64 \
  task.train.max_steps=200000 \
  model=cdm \
  model.arch=Perceiver
```

训练过程中会使用 `AffordanceDataset` 加载 3D 邻域点云与 3D 掩码，并 **不启用 diffusion 模块**。

- **评估**

```bash
python pipeline/step8_3d_training/test_no_diff.py \
  exp_dir=outputs/<your-exp-dir> \
  gpu=0
```

脚本会在对应 `exp_dir` 下找到最新的 checkpoint，对验证集进行 3D segmentation / contact 评估，并输出 `results.json`。

---

## 配置与实验管理

本项目使用 **Hydra + OmegaConf** 管理配置：

- 默认配置路径：`pipeline/step8_3d_training/configs/`（如果你在本地扩展，可以在此添加/修改实验配置）。
- 每次运行会自动在 `outputs/<date>_<time>_<exp_name>` 下创建实验目录，包含：
  - `log/runtime.log`：完整配置与训练日志
  - `ckpt/`：模型权重
  - `eval/`：评估结果

你可以通过修改命令行参数或配置文件，控制是否使用颜色、是否加载文本、点数、batch size 等。

---

## 贡献与扩展建议

- 如需扩展到其它场景数据集或机器人平台，可：
  - 实现新的 `DataParser` 或扩展 `AffordanceDataset`；
  - 在 `models/` 中新增 backbone 或注意力结构；
  - 增加新的 `pipeline/stepX_*` 以支持不同的多模态模型（例如替换 Qwen、Molmo、SAM2）。

- PR 建议附带：
  - 简要说明变更目的；
  - 对应示例命令或最小复现脚本。

---

## 许可证

根据你后续选择的开源协议填写，例如：

```text
MIT License

Copyright (c) 2025 ...
```

如需根据实际依赖/外部项目建议一个合适的 License（例如 MIT / Apache‑2.0 等），可以在开源前再做补充说明。

