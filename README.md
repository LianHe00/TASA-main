## [AAAI 2026]Task-Aware 3D Affordance Segmentation via 2D Guidance and Geometric Refinement

## Environment Setup

Python 3.10+ and a virtual environment are recommended.

```bash
git clone https://github.com/LianHe00/TASA-main.git
cd affseg
conda create -n affseg python=3.10
conda activate affseg
pip install -r requirements.txt
```

Notes:
- `pointops_cuda` (used by PointTransformer) may need to be built separately for your CUDA / PyTorch version (see PointTransformer official docs or project-specific notes).
- Weights for large models (Qwen, Molmo, SAM2, etc.) are not shipped with the repo; they are downloaded via `transformers` or official repositories when needed.

---

## Data Preparation (Brief)

We use data from the [Scenefun3D](https://scenefun3d.github.io) dataset. The downloaded data includes the following assets:

- **laser_scan_5mm** — dense 3D point clouds (5 mm resolution)
- **crop_mask** — segmentation masks used to crop the scene
- **annotations** — affordance / contact annotations
- **descriptions** — natural language descriptions per visit
- **hires_wide** — high-resolution wide-angle RGB frames
- **hires_wide_intrinsics** — camera intrinsics for the wide-angle frames
- **hires_depth** — depth maps aligned to the RGB frames
- **hires_poses** — camera poses (extrinsics) for 2D→3D projection

---

## Typical Usage

### 1. Text instructions → objects & affordances

```bash
python pipeline/step1_affordance/qwen.py \
  --input_instructions path/to/instructions.json \
  --output path/to/affordance.json
```

Output includes target object categories and affordance labels (e.g. door handle, drawer handle).

### 2. Frame selection & operation point prediction

- Use CLIP to select the top-K frames most relevant to the affordance:

```bash
python pipeline/step2_clipwithaffordance/clip_affordance.py \
  --frames_dir path/to/frames \
  --affordance_json path/to/affordance.json \
  --output path/to/topk_frames.json
```

- Use Qwen-VL to predict operation points on those frames:

```bash
python pipeline/step3_point_prediction/qwen_point.py \
  --frames path/to/topk_frames.json \
  --output path/to/points.json
```

### 3. 2D segmentation & 2D mask merging

```bash
# Segment with Molmo + SAM / SAM2
python pipeline/step5_molmo_sam/molmo_sam.py \
  --frames path/to/topk_frames.json \
  --points path/to/points.json \
  --output path/to/2d_masks

# Merge local masks back to full image coordinates
python pipeline/step6_molmo_merge/molmo_merge.py \
  --masks_dir path/to/2d_masks \
  --output path/to/merged_masks
```

### 4. 2D→3D lifting & point cloud extraction

```bash
python pipeline/step7_lift_3d/molmo_lift_2d_to_3d.py \
  --config path/to/lift_config.yaml \
  --masks path/to/merged_masks \
  --output path/to/3d_masks_and_pcd
```

This step uses geometry and parsing utilities from `scenefun3d_utils` to project 2D masks to 3D and produce local point clouds and 3D masks.

### 5. 3D training & evaluation (no diffusion)

- **Training (main entry)**

Run `train_no_diff.py` with Hydra; see `scripts/train.sh` for an example:

```bash
bash scripts/train.sh perceiver_division_8192_pointtransformer_with_new_loss
```

Or run directly:

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

Training uses `AffordanceDataset` to load 3D neighborhood point clouds and 3D masks; the **diffusion module is not used**.

- **Evaluation**

```bash
python pipeline/step8_3d_training/eval_no_diff.py \
  exp_dir=outputs/<your-exp-dir> \
  gpu=0
```

The script finds the latest checkpoint under `exp_dir`, runs 3D segmentation / contact evaluation on the validation set, and writes `results.json`.

---

## Configuration and Experiment Management

The project uses **Hydra + OmegaConf** for configuration:

- Default configs: `pipeline/step8_3d_training/configs/` (add or edit configs here for local experiments).
- Each run creates an experiment directory under `outputs/<date>_<time>_<exp_name>` with:
  - `log/runtime.log`: full config and training log
  - `ckpt/`: model checkpoints
  - `eval/`: evaluation outputs

You can override options (e.g. use color, load text, point count, batch size) via the command line or config files.

---

## Contributing and Extension

- To adapt to other scene datasets or robot platforms you can:
  - Implement a new `DataParser` or extend `AffordanceDataset`;
  - Add new backbones or attention modules under `models/`;
  - Add new `pipeline/stepX_*` steps for different multimodal models (e.g. replace Qwen, Molmo, SAM2).

- For pull requests, please include:
  - A short description of the change;
  - Example commands or a minimal reproduction script where relevant.

---

## Acknowledgement

This project is built on top of prior efforts in 3D functionality understanding. In particular, we took inspiration from **Fun3DU** ([GitHub repository](https://github.com/tev-fbk/fun3du)) for the overall pipeline design, and we use the **SceneFun3D** dataset ([project page](https://scenefun3d.github.io/)) as our main data source. We sincerely thank the authors of Fun3DU and SceneFun3D for releasing their code and dataset to the community.

## Citation

If you find this work helpful, please cite:

```text
@article{he2025task,
  title={Task-Aware 3D Affordance Segmentation via 2D Guidance and Geometric Refinement},
  author={He, Lian and Liu, Meng and Ye, Qilang and Zhou, Yu and Deng, Xiang and Ding, Gangyi},
  journal={arXiv preprint arXiv:2511.11702},
  year={2025}
}
```

---

## License

This project is released under the **Apache License 2.0**. See `LICENSE` for the full license text.
