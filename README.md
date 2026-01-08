# JustDepth
JustDepth: Real-Time Radar-Camera Depth Estimation with Single-Scan LiDAR Supervision

## Architecture
<p align="center">
  <img src="assets/JustDepth.png" alt="JustDepth Architecture" width="900"/>
</p>

## Results
<p align="center">
  <img src="assets/Results.png" alt="Results" width="900"/>
</p>

## Training
Multi-GPU training (torchrun):

~~~bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> torchrun --nproc_per_node=<NUM_GPUS> train.py
# Example:
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py
~~~

Single-GPU training:

~~~bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python train.py
# Example:
# CUDA_VISIBLE_DEVICES=0 python train.py
~~~

## Evaluation
Evaluate with a checkpoint:

~~~bash
python eval.py --checkpoint <PATH_TO_CKPT>
# Example:
# python eval.py --checkpoint /path/to/latest.ckpt
~~~