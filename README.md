<div align="center">

# JustDepth

### Real-Time Radar-Camera Depth Estimation With Single-Scan LiDAR Supervision

<p>
  <a href="https://ieeexplore.ieee.org/document/11358657">
    <img src="https://img.shields.io/badge/IEEE%20RA--L%20%E2%80%A2%20VOL.%2011%20NO.%203-00629B?style=for-the-badge&logo=ieee&logoColor=white" alt="IEEE RA-L">
  </a>
  &nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2607.22172">
    <img src="https://img.shields.io/badge/arXiv%20%E2%80%A2%202607.22172-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  &nbsp;&nbsp;
  <a href="https://youtu.be/EURdRO6OfS0">
    <img src="https://img.shields.io/badge/DEMO%20%E2%80%A2%20YOUTUBE-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Demo Video">
  </a>
  &nbsp;&nbsp;
  <a href="https://doi.org/10.1109/LRA.2026.3655274">
    <img src="https://img.shields.io/badge/DOI%20%E2%80%A2%2010.1109%2FLRA.2026.3655274-008CC1?style=for-the-badge&logo=doi&logoColor=white" alt="DOI">
  </a>
</p>

**A single-stage radar-camera depth estimator for real-time autonomous systems**

**Constant Latency · Single-Scan LiDAR Supervision · No Auxiliary Annotations**

<br>

[**Paper**](https://ieeexplore.ieee.org/document/11358657)
&nbsp;·&nbsp;
[**arXiv**](https://arxiv.org/abs/2607.22172)
&nbsp;·&nbsp;
[**Demo**](https://youtu.be/EURdRO6OfS0)
&nbsp;·&nbsp;
[**Checkpoints**](https://drive.google.com/drive/folders/176G2QK_zVTm5zYy4P9ZASQ2K0a4a23ny?usp=share_link)
&nbsp;·&nbsp;
[**Dataset Index Files**](https://drive.google.com/drive/folders/1WvbM3ydickJU4d3_7ahFWVZ8HLsYjZzo?usp=share_link)

</div>

---

## Overview

**JustDepth** is a real-time, single-stage radar-camera depth estimation framework for autonomous systems.

The network takes an RGB image and an automotive radar scan as input and directly predicts a dense metric depth map. It does not require an intermediate sparse or quasi-dense depth product, a pretrained monocular depth model, multi-frame LiDAR accumulation, or auxiliary annotations such as semantic masks and bounding boxes.

JustDepth compresses all radar returns in a frame into a fixed-width 1D representation. This design keeps the computation of the radar branch independent of the number of raw radar points. Image and radar features are fused through a **Height Fusion Block**, and a lightweight **Graph Neural Network** propagates depth cues across the scene.

A **training-only Confidence Decoder** provides direct supervision to the fusion module by learning radar-supported pixels. This decoder is discarded during inference and therefore adds no test-time latency.

To reduce stripe artifacts caused by **LiDAR Distribution Leakage**, JustDepth uses point upsampling and synchronized rotation augmentation with reflection padding. The paper also introduces the **Vertical-Horizontal Gradient Ratio**, or **VHGR**, to quantify scanline artifacts across the predicted depth map.

---

## Highlights

<table>
  <tr>
    <td><b>⚡ Single Stage</b></td>
    <td>Direct dense depth prediction without intermediate depth products</td>
  </tr>
  <tr>
    <td><b>🏎️ Real Time</b></td>
    <td>14.8 ms per frame with the 8-layer GNN model on an NVIDIA RTX 4070 Ti</td>
  </tr>
  <tr>
    <td><b>📡 Radar-Camera Fusion</b></td>
    <td>Dense metric depth estimation from an RGB image and automotive radar</td>
  </tr>
  <tr>
    <td><b>📏 Constant Latency</b></td>
    <td>Fixed-width 1D radar encoding with computation independent of radar point count</td>
  </tr>
  <tr>
    <td><b>📍 Single-Scan LiDAR Supervision</b></td>
    <td>Training with one LiDAR sweep instead of accumulated multi-frame LiDAR</td>
  </tr>
  <tr>
    <td><b>🚫 No Auxiliary Annotations</b></td>
    <td>No semantic masks, panoptic masks, or bounding boxes</td>
  </tr>
  <tr>
    <td><b>🧠 Training-Only Confidence Decoder</b></td>
    <td>Improves radar-supported feature learning with zero test-time overhead</td>
  </tr>
  <tr>
    <td><b>🌐 Global Depth Propagation</b></td>
    <td>Lightweight GNN propagation over image feature tokens</td>
  </tr>
  <tr>
    <td><b>🧹 LDL Mitigation</b></td>
    <td>Point upsampling and synchronized rotation reduce scanline artifacts</td>
  </tr>
  <tr>
    <td><b>📊 VHGR Metric</b></td>
    <td>Gradient-based evaluation of LiDAR Distribution Leakage artifacts</td>
  </tr>
</table>

---

## Demo Video

<div align="center">

<a href="https://youtu.be/EURdRO6OfS0">
  <img src="https://img.youtube.com/vi/EURdRO6OfS0/maxresdefault.jpg" alt="JustDepth Demo Video" width="950">
</a>

<br>

<a href="https://youtu.be/EURdRO6OfS0">
  <img src="https://img.shields.io/badge/WATCH%20THE%20FULL%20DEMO-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Watch the Demo">
</a>

</div>

---

## Architecture

<div align="center">

<img src="assets/JustDepth.png" alt="JustDepth Architecture" width="950">

</div>

JustDepth consists of six main components:

1. **Image Encoder**  
   A ResNet-style backbone extracts hierarchical image features from the RGB input.

2. **Fixed-Width Radar Encoder**  
   Calibrated radar points are projected onto the image plane and compressed into a fixed-width 1D scan. The minimum range is retained for each image column.

3. **Height Fusion Block**  
   Radar and image features are fused column by column through height-wise self-attention.

4. **Graph-Based Global Propagation**  
   A lightweight GNN builds a feature-space K-NN graph and propagates depth information between related image locations.

5. **Depth Decoder**  
   A U-Net-style decoder combines the globally propagated features with intermediate image features to produce a dense depth map.

6. **Training-Only Confidence Decoder**  
   An auxiliary decoder predicts radar-supported pixels during training and is removed during inference.

---

## Main Contributions

### 1. Single-Stage, Constant-Latency Architecture

All radar returns are encoded into a fixed-width 1D representation. The radar encoder therefore has effectively constant computation with respect to the number of radar returns.

The encoded radar features are fused with image features through height-wise self-attention. A lightweight GNN then propagates depth cues globally before the final dense depth prediction.

### 2. Training-Only Confidence Decoder

The Confidence Decoder directly supervises the fusion module to identify pixels supported by radar measurements.

Unlike methods that first generate an explicit intermediate radar depth map, the confidence branch is used only as an auxiliary training signal. It is completely discarded during inference and introduces zero test-time overhead.

### 3. LiDAR Distribution Leakage Mitigation

Single-scan LiDAR supervision can cause the model to reproduce the horizontal scanline structure of the LiDAR sensor.

JustDepth mitigates these artifacts using:

- Point upsampling between compatible LiDAR samples
- Synchronized rotation of the RGB image, radar, and LiDAR
- Reflection padding for rotated RGB images
- Edge-aware smoothness regularization

The paper also introduces **VHGR** to measure the imbalance between vertical and horizontal depth gradients caused by stripe artifacts.

---

## Results

<div align="center">

<img src="assets/Results.png" alt="JustDepth Qualitative Results" width="950">

</div>

JustDepth produces complete dense depth maps across diverse daytime and nighttime nuScenes scenes while reducing the horizontal stripe artifacts commonly associated with sparse single-scan LiDAR supervision.

---

## Runtime vs Accuracy

<div align="center">

<img src="assets/PerfGraph.png" alt="JustDepth Runtime and Accuracy Trade-Off" width="950">

</div>

The 8-layer GNN configuration processes one frame in approximately **14.8 ms** on an NVIDIA RTX 4070 Ti.

In the paper's nuScenes comparison, JustDepth maintains competitive depth accuracy while reducing inference time by **39.7×** relative to GET-UP.

---

## Benchmark

<div align="center">

<img src="assets/PerfTable.png" alt="JustDepth Benchmark Table" width="950">

</div>

The reported results use images with a resolution of **900 × 1600** and a maximum evaluation depth of **80 m**.

| Property | JustDepth |
|:---|:---|
| Input | RGB image and automotive radar |
| Output | Dense metric depth map |
| LiDAR supervision | Single scan |
| Auxiliary annotations | None |
| Intermediate depth product | None |
| Pretrained monocular depth module | None |
| Inference device | NVIDIA RTX 4070 Ti |
| Inference time | 14.8 ms |
| GNN layers | 8 |
| Radar sweeps | 1 |
| Image frames | 1 |

---

## Supported Datasets

<table>
  <tr>
    <td><b>nuScenes</b></td>
    <td>✅ Supported</td>
  </tr>
  <tr>
    <td><b>ZJU-4DRadarCam</b></td>
    <td>✅ Supported</td>
  </tr>
</table>

---

## Downloads

<div align="center">

<a href="https://drive.google.com/drive/folders/1WvbM3ydickJU4d3_7ahFWVZ8HLsYjZzo?usp=share_link">
  <img src="https://img.shields.io/badge/DOWNLOAD%20DATASET%20INDEX%20FILES-4285F4?style=for-the-badge&logo=googledrive&logoColor=white" alt="Download Dataset Index Files">
</a>

&nbsp;&nbsp;

<a href="https://drive.google.com/drive/folders/176G2QK_zVTm5zYy4P9ZASQ2K0a4a23ny?usp=share_link">
  <img src="https://img.shields.io/badge/DOWNLOAD%20PRETRAINED%20CHECKPOINTS-34A853?style=for-the-badge&logo=googledrive&logoColor=white" alt="Download Pretrained Checkpoints">
</a>

</div>

---

## Data Layout

Place the datasets under `data/`, or edit the corresponding paths in `configs/*.txt`.

```text
JustDepth/
├── data/
│   ├── nuscenes/
│   │   └── samples/
│   ├── nuscenes_radar_5sweeps_infos_train.pkl
│   ├── nuscenes_radar_5sweeps_infos_test.pkl
│   └── zju/
│       ├── train.txt
│       ├── test.txt
│       ├── image/
│       ├── gt/
│       └── radar/
├── configs/
├── train.py
├── eval.py
└── save_confidence_map.py
```

---

## Installation

### Requirements

- Python 3.11.13
- CUDA-compatible GPU
- PyTorch

### Setup

```bash
# Create a clean environment
conda create -n justdepth python=3.11.13 -y

# Activate the environment
conda activate justdepth

# Install dependencies
pip install -r requirements.txt
```

---

## Confidence Maps

Training uses binary confidence maps as targets for the training-only Confidence Decoder.

The maps can be precomputed before training to avoid generating them repeatedly inside the data loader.

### nuScenes

```bash
python save_confidence_map.py \
  --dataset nuscenes \
  --nuscenes-path data/nuscenes_radar_5sweeps_infos_train.pkl \
  --nuscenes-root data/nuscenes/samples \
  --rule column \
  --rid-outliers \
  --link-lidar \
  --output-dir confidence_map/nuscenes_train \
  --workers 8
```

For nuScenes, use the column confidence rule with LiDAR linking and outlier removal enabled.

### ZJU-4DRadarCam

```bash
python save_confidence_map.py \
  --dataset zju \
  --zju-path data/zju/train.txt \
  --zju-root data/zju \
  --rule dot \
  --output-dir confidence_map/zju_train \
  --workers 8
```

---

## Training

### nuScenes

#### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> torchrun \
  --nproc_per_node=<NUM_GPUS> \
  train.py \
  --config configs/nuscenes_train.txt
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  train.py \
  --config configs/nuscenes_train.txt
```

#### Single-GPU Training

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python train.py \
  --config configs/nuscenes_train.txt \
  --local
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/nuscenes_train.txt \
  --local
```

### ZJU-4DRadarCam

#### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> torchrun \
  --nproc_per_node=<NUM_GPUS> \
  train.py \
  --config configs/zju_train.txt
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  train.py \
  --config configs/zju_train.txt
```

#### Single-GPU Training

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python train.py \
  --config configs/zju_train.txt \
  --local
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/zju_train.txt \
  --local
```

---

## Evaluation

### nuScenes

```bash
python eval.py \
  --config configs/nuscenes_eval.txt \
  --checkpoint <PATH_TO_CKPT>
```

Example:

```bash
python eval.py \
  --config configs/nuscenes_eval.txt \
  --checkpoint train_log/models/latest.ckpt
```

### ZJU-4DRadarCam

```bash
python eval.py \
  --config configs/zju_eval.txt \
  --checkpoint <PATH_TO_CKPT>
```

Example:

```bash
python eval.py \
  --config configs/zju_eval.txt \
  --checkpoint train_log/models/latest.ckpt
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{yun2026justdepth,
  title={JustDepth: Real-Time Radar-Camera Depth Estimation With Single-Scan LiDAR Supervision},
  author={Yun, Wooyung and Kim, Dongwook and Lee, Soomok},
  journal={IEEE Robotics and Automation Letters},
  year={2026},
  volume={11},
  number={3},
  pages={2770--2777},
  doi={10.1109/LRA.2026.3655274}
}
```

---

<div align="center">

<br>

<a href="https://github.com/TPyun/JustDepth">
  <img src="https://img.shields.io/badge/⭐%20STAR%20THIS%20REPOSITORY-181717?style=for-the-badge&logo=github&logoColor=white" alt="Star JustDepth">
</a>

<br><br>

</div>
