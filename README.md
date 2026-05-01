# Rep3D: Spatially-Adaptive Gradient Re-parameterization for 3D Large Kernel Optimization

<p align="center">
  <a href="https://icml.cc/"><img src="https://img.shields.io/badge/ICML-2026-blue.svg" alt="ICML 2026"></a>
  <a href="https://arxiv.org/abs/2505.19603"><img src="https://img.shields.io/badge/arXiv-2505.19603-b31b1b.svg" alt="arXiv"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://monai.io/"><img src="https://img.shields.io/badge/MONAI-0.9%2B-00A1E0.svg" alt="MONAI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

This is the official PyTorch implementation of our **ICML 2026** paper:

> **Spatially-Adaptive Gradient Re-parameterization for 3D Large Kernel Optimization**
> Ho Hin Lee, Quan Liu, Shunxing Bao, Yuankai Huo, Bennett A. Landman
> *International Conference on Machine Learning (ICML), 2026 (Accepted)*
> [[Paper]](https://arxiv.org/abs/2505.19603)

> **TL;DR.** Naively scaling up 3D convolutional kernels (e.g., 21×21×21) does *not* improve segmentation — performance saturates because the optimization dynamics across kernel elements are non-uniform: central elements converge fast, peripheral ones lag. **Rep3D** introduces a lightweight, learnable spatial bias derived from effective receptive field (ERF) theory that adaptively re-weights kernel updates during training. The result is a plain, single-branch encoder that consistently outperforms transformer- and CNN-based SOTA on five 3D medical segmentation benchmarks — and the modulation generator is **dropped at inference** for zero deployment overhead.

---

## Table of Contents

- [Highlights](#highlights)
- [Method Overview](#method-overview)
- [Main Results](#main-results)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Repository Structure](#repository-structure)
- [Pretrained Weights](#pretrained-weights)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Highlights

- **State-of-the-art on 5 benchmarks.** AMOS-CT (0.910 Dice), AMOS-MRI (0.864), KiTS (0.736), MSD Pancreas (0.723), MSD Hepatic Vessel (0.674) — outperforming SwinUNETR, UNesT-B, 3D UX-Net, MedNeXt, STU-Net-H, ResEnc nnU-Net, and others (p < 0.01).
- **Theoretically grounded.** We derive that CSLA-style structural re-parameterization implicitly induces a *spatially varying learning-rate field* (faster at the kernel center, slower at the periphery) under both SGD and AdamW. Rep3D makes this implicit dynamic explicit and learnable.
- **Plain encoder, single branch.** No multi-branch CSLA, no parallel small-kernel paths at training. Just a 21×21×21 depthwise convolution per block, modulated by an LRBM mask.
- **Zero-cost inference.** The modulation generator is needed only during training. Set `deploy=True` to drop it entirely — the deployed model is a plain DWConv → BN → GELU stack.
- **Plug-and-play.** LRBM also boosts the existing 3D UX-Net backbone (0.890 → 0.897 mean Dice on AMOS-CT) without architectural changes.

## Method Overview

<p align="center">
  <em>Place architecture/Figure 1 from the paper here, e.g.:</em><br>
  <code>&lt;img src="assets/rep3d_overview.png" width="85%"&gt;</code>
</p>

**The core idea.** For a CSLA block combining a large kernel `W_L` and a small kernel `W_S` with scales `α_L`, `α_S`, the equivalent kernel `W' = α_L W_L + α_S W_S` receives spatially non-uniform gradient contributions. Because `W_S` only overlaps the central region of `W'`, the central elements receive gradient signal from *both* branches while the periphery receives signal only from `W_L`. This produces an effective element-wise learning-rate field:

```
λ_eff(Δx) = α_L λ_L                  (peripheral offsets)
            α_L λ_L + α_S λ_S        (central offsets)
```

This pattern mirrors the local-to-global diffusion of effective receptive fields (ERFs). Rep3D translates this implicit dynamic into an **explicit, learnable spatial bias** on a single large-kernel branch.

**Low-Rank Receptive Bias Modeling (LRBM).** We construct a reciprocal distance-decay prior `P` centered on the kernel and let a tiny 2-layer depthwise generator `f_θ` learn an additive correction:

```
P[k]    = β / (||k − c||₂ + β)            distance-decay prior
M       = P + f_θ(P)                       learnable modulation mask
W_eff   = W ⊙ M                            element-wise re-parameterization
```

The generator `f_θ` is two depthwise 3D convolutions (kernel size 7) with LayerNorm and a sigmoid gate — a few thousand extra parameters per block. At inference, `f_θ` is discarded; only the modulated `W` is deployed.

**Architecture.** The Rep3D backbone is a 4-stage hierarchical encoder (channels `[48, 96, 192, 384]`, depths `[2, 2, 2, 2]`) with a stem that uses a 7×7×7 stride-2 convolution. Each Rep3D block is a single depthwise 21×21×21 convolution + BN + GELU, modulated by LRBM during training. The decoder uses MONAI's `UnetrUpBlock` with skip connections, mirroring 3D UX-Net.

## Main Results

All scores are mean Dice (DSC).

### Tissue & tumor segmentation (KiTS, MSD Pancreas, MSD Hepatic Vessel)

| Method | #Params | KiTS Mean | Pancreas Mean | Hepatic Mean |
|---|---|---|---|---|
| 3D U-Net | 4.81M | 0.645 | 0.648 | 0.589 |
| nn-UNet | 31.2M | 0.706 | 0.703 | 0.660 |
| UNETR | 92.8M | 0.648 | 0.667 | 0.590 |
| nnFormer | 149.3M | 0.664 | 0.686 | 0.613 |
| SwinUNETR | 62.2M | 0.680 | 0.708 | 0.635 |
| 3D UX-Net (k=7) | 53.0M | 0.697 | 0.676 | 0.652 |
| UNesT-B | 87.2M | 0.710 | 0.690 | 0.640 |
| **Rep3D (Fixed Prior)** | 65.8M | 0.727 | 0.715 | 0.658 |
| **Rep3D (Ours)** | 66.0M | **0.736*** | **0.723*** | **0.674*** |

*p < 0.01, paired Wilcoxon signed-rank test against all baselines.

### Multi-organ segmentation (AMOS, train-from-scratch)

| Method | AMOS-CT (Avg) | AMOS-MRI (Avg) |
|---|---|---|
| nn-UNet (1000 ep) | 0.887 | 0.847 |
| SwinUNETR | 0.871 | 0.836 |
| 3D UX-Net (k=7) | 0.890 | 0.841 |
| 3D UX-Net (k=21) | 0.891 | 0.840 |
| UNesT-B | 0.891 | 0.843 |
| RepOptimizer | 0.892 | 0.847 |
| **Rep3D (Fixed Prior)** | 0.902 | 0.855 |
| **Rep3D (Ours)** | **0.910*** | **0.864*** |

### Comparisons with recent nnU-Net variants (full 1000-epoch schedules)

| Method | AMOS-CT | AMOS-MRI | KiTS | Pancreas | Hepatic |
|---|---|---|---|---|---|
| nnU-Net | 0.887 | 0.847 | 0.706 | 0.703 | 0.660 |
| ResEnc nnU-Net | 0.892 | 0.850 | 0.711 | 0.706 | 0.661 |
| STU-Net-H | 0.900 | 0.848 | 0.707 | 0.712 | 0.648 |
| MedNeXt | 0.897 | 0.856 | 0.720 | 0.713 | 0.663 |
| **Rep3D (Ours)** | **0.910** | **0.864** | **0.736** | **0.723** | **0.674** |

See the paper for per-organ breakdowns and all ablations (generator depth, generator kernel size, vanilla baseline, 3D UX-Net + LRBM).

---

## Installation

We tested with **Python 3.9**, **PyTorch 1.12.0**, **CUDA 11.3**, and a single **NVIDIA A100 (40GB / 80GB)**. Other configurations should work but have not been verified.

```bash
# 1. Clone the repository
git clone https://github.com/leeh43/Rep3D.git
cd Rep3D

# 2. Create a conda environment
conda create -n rep3d python=3.9 -y
conda activate rep3d

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 4. Install MONAI and the rest
pip install monai==0.9.0
pip install -r requirements.txt
```

`requirements.txt` should include at least: `nibabel`, `einops`, `tensorboard`, `tqdm`, `batchgenerators`, `numpy`, `scipy`.

> **A note on the 21³ depthwise kernel.** Native PyTorch handles 3D depthwise convolutions of this size, but training is memory-bound. We recommend AMP (`torch.cuda.amp`) and a 40GB+ GPU for batch size ≥ 2. Lower the `--cache_rate` if you hit RAM limits.

## Data Preparation

Rep3D currently supports five datasets, each expected in the same MONAI-style folder layout:

```
<root>/
├── imagesTr/      # training volumes (.nii.gz)
├── labelsTr/      # training labels  (.nii.gz)
├── imagesVal/     # validation volumes
└── labelsVal/     # validation labels
```

Public dataset sources:

- **AMOS22** — [https://amos22.grand-challenge.org/](https://amos22.grand-challenge.org/) (CT: 200 scans, 15 organs; MRI: 33 scans, 13 organs)
- **KiTS21** — [https://kits21.kits-challenge.org/](https://kits21.kits-challenge.org/) (210 CT scans; kidney, tumor, cyst)
- **MSD Pancreas** & **MSD Hepatic Vessel** — [http://medicaldecathlon.com/](http://medicaldecathlon.com/)

The dataset string passed via `--dataset` selects the preprocessing pipeline (intensity window, voxel spacing, # classes). Currently supported values: `amos`, `amos_mri`, `kits`, `pancreas`, `hepatic`. Preprocessing is applied on-the-fly via MONAI transforms in `load_datasets_transforms.py`:

| Dataset | Voxel Spacing (mm) | Intensity Window | # Classes |
|---|---|---|---|
| AMOS-CT | 1.5 × 1.5 × 2.0 | [-125, 275] | 16 |
| AMOS-MRI | 1.0 × 1.0 × 1.0 | [0, 1000] | 14 |
| KiTS | 1.5 × 1.5 × 2.0 | [-125, 275] | 4 |
| MSD Pancreas | 1.5 × 1.5 × 2.0 | [-125, 275] | 3 |
| MSD Hepatic Vessel | 1.5 × 1.5 × 2.0 | [0, 230] | 3 |

Common patch settings (training): random foreground crops of `96 × 96 × 96`, two sub-volumes per subject, augmentations with rotation (±π/30), intensity shift (0.10), and isotropic scaling (0.1).

## Training

Train Rep3D from scratch on AMOS-CT:

```bash
python main_train.py \
    --root /path/to/amos_ct \
    --output ./runs/rep3d_amos_ct \
    --dataset amos \
    --network REP3D \
    --batch_size 1 \
    --crop_sample 2 \
    --lr 1e-4 \
    --optim AdamW \
    --max_iter 60000 \
    --eval_step 500 \
    --gpu 0 \
    --cache_rate 0.1 \
    --num_workers 2
```

Training reproduces the paper's setup: 60,000 iterations on a single A100, AdamW (β=(0.9, 0.999), ε=1e-8, weight decay 0.08), peak LR 1e-4, dual sliding-window crops at 96³, validation every 500 steps. The best checkpoint (`best_metric_model.pth`) is saved to `--output` and TensorBoard logs go to `<output>/tensorboard/`.

**Switching datasets.** Pass `--dataset {amos, amos_mri, kits, pancreas, hepatic}`. The number of output classes is set automatically.

**Switching backbones.** The training script also includes the baselines used in the paper. Use `--network {REP3D, 3DUXNET, SwinUNETR, UNETR, nnFormer, TransBTS}`.

**Resuming from a checkpoint.**

```bash
python main_train.py \
    --root /path/to/data --output ./runs/exp --dataset amos --network REP3D \
    --pretrain True --pretrained_weights ./runs/exp/best_metric_model.pth
```

**Approximate per-epoch training time** (single A100, batch size 1):

| Dataset | Time / epoch |
|---|---|
| AMOS-CT | ~7 min |
| AMOS-MRI | ~1 min |
| KiTS | ~9 min |
| MSD Pancreas | ~5 min |
| MSD Hepatic Vessel | ~12 min |

## Inference

Inference uses MONAI's sliding-window inferer. Rep3D is constructed with `deploy=True` so the LRBM generator is bypassed — the deployed model is a plain `DWConv-21 → BN → GELU` stack:

```bash
python test_seg.py \
    --root /path/to/amos_ct \
    --output ./predictions/rep3d_amos_ct \
    --dataset amos \
    --network REP3D \
    --trained_weights ./runs/rep3d_amos_ct/best_metric_model.pth \
    --sw_batch_size 4 \
    --overlap 0.5 \
    --gpu 0
```

Predictions are written to `--output` as `*_seg.nii.gz` files (resampled back to each volume's native space via MONAI's `Invertd`).

## Repository Structure

```
Rep3D/
├── main_train.py                  # Training entry point (REP3D + all baselines)
├── test_seg.py                    # Sliding-window inference entry point
├── load_datasets_transforms.py    # MONAI data loaders & transforms for all 5 datasets
├── networks/
│   ├── Rep3D/
│   │   ├── network_backbone.py    # REP3D encoder–decoder wrapper (UnetrBasicBlock + UnetrUpBlock)
│   │   └── rep3d_encoder.py       # Rep3D blocks, LRBM generator, distance-decay prior
│   ├── UXNet_3D/                  # 3D UX-Net baseline
│   ├── nnFormer/                  # nnFormer baseline
│   └── TransBTS/                  # TransBTS baseline
├── requirements.txt
└── README.md
```

The two files most worth reading:

- **`networks/Rep3D/rep3d_encoder.py`** — `compute_distance_prior()` builds the reciprocal distance-decay map; `rep3d_block` applies the 2-layer depthwise generator + sigmoid gate to produce the modulation mask `M = P + f_θ(P)` and re-parameterizes the kernel weight in-place during training (skipped when `deploy=True`).
- **`networks/Rep3D/network_backbone.py`** — Wires the four Rep3D stages (depths `[2,2,2,2]`, channels `[48,96,192,384]`) into a U-shaped encoder–decoder using MONAI's `UnetrBasicBlock` for skip-paths and `UnetrUpBlock` for upsampling.

## Pretrained Weights

Pretrained checkpoints will be released here after the camera-ready deadline:

| Dataset | Model | Mean Dice | Download |
|---|---|---|---|
| AMOS-CT | Rep3D (LRBM) | 0.910 | _coming soon_ |
| AMOS-MRI | Rep3D (LRBM) | 0.864 | _coming soon_ |
| KiTS | Rep3D (LRBM) | 0.736 | _coming soon_ |
| MSD Pancreas | Rep3D (LRBM) | 0.723 | _coming soon_ |
| MSD Hepatic Vessel | Rep3D (LRBM) | 0.674 | _coming soon_ |

## Citation

If Rep3D or our analysis of spatially-varying convergence in re-parameterized convolutions is useful in your work, please cite:

```bibtex
@inproceedings{lee2026rep3d,
  title     = {Spatially-Adaptive Gradient Re-parameterization for 3D Large Kernel Optimization},
  author    = {Lee, Ho Hin and Liu, Quan and Bao, Shunxing and Huo, Yuankai and Landman, Bennett A.},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

You may also be interested in our prior work on this line:

```bibtex
@inproceedings{lee2023uxnet,
  title     = {{3D UX-Net}: A Large Kernel Volumetric ConvNet Modernizing Hierarchical Transformer for Medical Image Segmentation},
  author    = {Lee, Ho Hin and Bao, Shunxing and Huo, Yuankai and Landman, Bennett A.},
  booktitle = {The Eleventh International Conference on Learning Representations (ICLR)},
  year      = {2023}
}

@article{lee2023repuxnet,
  title   = {Scaling Up {3D} Kernels with {Bayesian} Frequency Re-parameterization for Medical Image Segmentation},
  author  = {Lee, Ho Hin and Liu, Quan and Bao, Shunxing and Yang, Qi and Yu, Xin and Cai, Leon Y. and Li, Thomas and Huo, Yuankai and Koutsoukos, Xenofon and Landman, Bennett A.},
  journal = {arXiv preprint arXiv:2303.05785},
  year    = {2023}
}
```

## Acknowledgments

The codebase builds on several excellent open-source projects, and we thank their authors:

- **[3D UX-Net](https://github.com/MASILab/3DUX-Net)** — encoder–decoder backbone and training pipeline
- **[MONAI](https://monai.io/)** — transforms, sliding-window inference, `UnetrBasicBlock` / `UnetrUpBlock`
- **[RepLKNet](https://github.com/DingXiaoH/RepLKNet-pytorch)** and **[RepOptimizer](https://github.com/DingXiaoH/RepOptimizers)** — the structural- and gradient-re-parameterization viewpoints we extend
- **[SwinUNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR)**, **[nnFormer](https://github.com/282857341/nnFormer)**, **[TransBTS](https://github.com/Wenxuan-1119/TransBTS)** — strong baselines used in our comparisons

This work was supported by Vanderbilt University and the Medical-image Analysis and Statistical Interpretation (MASI) Laboratory.

## Contact

For questions about the paper or the code, please open a [GitHub issue](https://github.com/leeh43/Rep3D/issues) or contact:

**Ho Hin Lee** — `ho.hin.lee@vanderbilt.edu`

---

<sub>Released under the MIT License. See [LICENSE](LICENSE) for details.</sub>