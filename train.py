import argparse
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import dataset 
import timm
from timm.models import create_model
import model


from utils import (
    expand_lidar_points,
)

from loguru import logger
logger.remove()
 
# -----------------------------
# 환경 변수 (분산 학습을 위한)
# -----------------------------
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

if LOCAL_RANK == 0:
    logger.add(sys.stdout, colorize=True, level="INFO", 
        format="<green>[{time:%m-%d %H:%M:%S}]</green> {message}")
else:
    logger.add(sys.stderr, colorize=True, level="ERROR", 
        format="<green>[{time:%m-%d %H:%M:%S}]</green> {message}")


def param_groups_weight_decay(model: nn.Module,
                              weight_decay: float = 1e-4,
                              norm_weight_decay: float = 0.0,
                              bias_weight_decay: float = 0.0,
                              skip_names: tuple = ()) -> list:
    norm_layers = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.LocalResponseNorm
    )

    decay_params = []
    norm_params  = []
    bias_params  = []
    other_nodecay = []

    # 중복 방지
    seen = set()
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if full_name in seen:
                continue
            seen.add(full_name)

            if full_name.endswith(".bias"):
                bias_params.append(param)
            elif isinstance(module, norm_layers):
                norm_params.append(param)
            elif any(full_name.startswith(s) or full_name.endswith(s) for s in skip_names):
                other_nodecay.append(param)
            else:
                # 일반적으로 Conv/Linear/Attention weight
                decay_params.append(param)

    param_groups = []
    if len(decay_params):
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if len(norm_params):
        param_groups.append({"params": norm_params, "weight_decay": norm_weight_decay})
    if len(bias_params):
        param_groups.append({"params": bias_params, "weight_decay": bias_weight_decay})
    if len(other_nodecay):
        param_groups.append({"params": other_nodecay, "weight_decay": 0.0})

    return param_groups

class TrainClock:
    def __init__(self):
        self.epoch = 0
        self.step = 0
    
    def tick(self):
        self.step += 1

    def tock(self):
        self.epoch += 1

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'step': self.step
        }
    
    def restore_checkpoint(self, ckp):
        self.epoch = ckp['epoch']
        self.step = ckp['step']

def ensure_dir(path: Path):
    path = Path(path)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)

def format_time(elapsed):
    elapsed = int(elapsed)
    h = elapsed // 3600
    m = elapsed % 3600 // 60
    s = elapsed % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# -----------------------------
# config
# -----------------------------
class config:
    base_lr = 0.001
    epoch_num = 200
    warmup_epochs = 0
    checkpoint_interval = 1
    log_interval = 200

    exp_dir = os.path.dirname(__file__)
    exp_name = os.path.basename(exp_dir)
    local_train_log_path = './train_log'
    log_dir = str(local_train_log_path)
    log_model_dir = os.path.join(local_train_log_path, 'models')
    
    
    # DataLoader 설정
    params = {
        'batch_size': 8,
        'num_workers': 8,
        'persistent_workers': True,
        'prefetch_factor': 2,
        'pin_memory': True,
    }

# -----------------------------
# Session 클래스
# -----------------------------
class Session:
    def __init__(self, config, net=None, rank=0, local_rank=0):
        self.log_dir = config.log_dir
        ensure_dir(self.log_dir)
        self.model_dir = config.log_model_dir
        ensure_dir(self.model_dir)

        self.clock = TrainClock()
        self.config = config
        
        self.net = net
        self.optimizer = None
        self.lr_scheduler = None
        
        self.rank = rank
        self.local_rank = local_rank

    def start(self):
        self.save_checkpoint('start')

    def save_checkpoint(self, name):
        """체크포인트 저장"""
        if self.rank != 0:
            return
        net = self.net.module if isinstance(self.net, DDP) else self.net
        
        ckp = {
            'network': net.state_dict(),
            'clock': self.clock.make_checkpoint(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }
        torch.save(ckp, Path(self.model_dir) / (name+'.ckpt'))
        logger.info(f"Saved checkpoint at {name}")

    def load_misc_checkpoint(self, ckp_path:Path):
        """optimizer, scheduler, clock 등 부수 checkpoint 로드"""
        checkpoint = torch.load(
            ckp_path, map_location=torch.device(f"cuda:{self.local_rank}")
        )
        if 'clock' in checkpoint:
            self.clock.restore_checkpoint(checkpoint['clock'])
        if 'optimizer' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        if self.lr_scheduler:
            self.lr_scheduler.last_epoch = self.clock.step
            print(f"Resuming from step {self.clock.step}, epoch {self.clock.epoch}")
            self.lr_scheduler.step()

        net = self.net.module if isinstance(self.net, DDP) else self.net
        net.load_state_dict(checkpoint['network'], strict=True)
            
        print(f'Latest Epoch: {self.clock.epoch}, Latest Step: {self.clock.step}')
        logger.info(f"Loaded checkpoint from {ckp_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart', action='store_true', help='Restart from checkpoint')
    parser.add_argument('--local', action='store_true', help='Local mode (disable distributed)')
    parser.add_argument('--model-name', default='justdepth', type=str, help='Name of timm model')
    args = parser.parse_args()

    # log 파일 설정
    if LOCAL_RANK == 0:
        log_path = Path(config.log_dir) / "worklog.log"
        logger.add(str(log_path.resolve()), colorize=True, level="INFO", 
            format="<green>[{time:%m-%d %H:%M:%S}]</green> {message}")
    else:
        log_path = Path(config.log_dir) / f"worklog_{RANK}.log"
        logger.add(str(log_path.resolve()), colorize=True, level="ERROR", 
            format="<green>[{time:%m-%d %H:%M:%S}]</green> {message}")

    # 분산 학습 초기화
    if not args.local:
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(backend='nccl')
    device_id = LOCAL_RANK if not args.local else 0

    # -----------------------------
    # timm 모델 생성
    # -----------------------------
    net = create_model(
        model_name=args.model_name,
    ).cuda(device_id)
    
    if (torch.cuda.is_available() 
        and torch.cuda.device_count() > 1 
        and not args.local):
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    
    # Session 생성
    sess = Session(config, net=net, rank=RANK, local_rank=LOCAL_RANK)
    
    # DDP 감싸기
    if (torch.cuda.is_available() 
        and torch.cuda.device_count() > 1 
        and not args.local):
        
        logger.info("Using DDP train Model!")
        net = DDP(sess.net, device_ids=[device_id], output_device=device_id, find_unused_parameters=False)
        sess.net = net

    # -----------------------------
    # 데이터셋 및 DataLoader
    # -----------------------------
    depth_dataset = dataset.RCDepthDataset(
        link_lidar=True,
        rid_outliers=True,
        augmentation=True,
        rotation=True,
    )
    train_sampler = DistributedSampler(depth_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
    train_loader = DataLoader(depth_dataset, **sess.config.params, sampler=train_sampler, drop_last=True)
    
    # -----------------------------
    # Optimizer / LR Scheduler
    # -----------------------------
    model_body = sess.net.module if hasattr(sess.net, "module") else sess.net

    optim_groups = param_groups_weight_decay(model_body, weight_decay=1e-3)

    opt = torch.optim.AdamW(
        optim_groups,
        lr=config.base_lr,
        betas=(0.9, 0.999),
        fused=True,         # CUDA 11.6+ / Ampere↑에서 가속
    )
    total_step = len(train_loader) * config.epoch_num

    # 추가: warmup 설정
    warmup_steps = config.warmup_epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return 0.45 * (1. + np.cos(np.pi * (step - warmup_steps) / (total_step - warmup_steps))) + 0.1
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    
    sess.optimizer = opt
    sess.lr_scheduler = lr_scheduler
    
    # -----------------------------
    # Load Checkpoint(optional)
    # -----------------------------
    continue_path = Path(os.path.join(config.log_model_dir, "latest.ckpt")) if args.restart else None
    if continue_path and continue_path.exists():
        sess.load_misc_checkpoint(continue_path)
        
    
    # -----------------------------
    # Training
    # -----------------------------
    sess.start()
    
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    huber_loss = nn.SmoothL1Loss(reduction='none')

    _SOBEL_X = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3) / 4
    _SOBEL_Y = _SOBEL_X.transpose(-1,-2).contiguous()

    def image_weight_map(image):
        gray = image.mean(dim=1, keepdim=True)
        sobel_x = _SOBEL_X.to(image.device)
        sobel_y = _SOBEL_Y.to(image.device)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        return torch.exp(-grad_mag)

    def sobel_smoothness_loss(pred_depth, image):
        B = pred_depth.size(0)
        dx = torch.abs(pred_depth[..., :, 1:] - pred_depth[..., :, :-1])    # (B,1,H,W-1)
        dy = torch.abs(pred_depth[..., 1:, :] - pred_depth[..., :-1, :])    # (B,1,H-1,W)
        w = image_weight_map(image)                                         # (B,1,H,W)
        wx = w[..., :, 1:]                                                  # (B,1,H,W-1)
        wy = w[..., 1:, :]                                                  # (B,1,H-1,W)
        smooth_x = (dx * wx).view(B, -1).mean(dim=1)                         # (B,)
        smooth_y = (dy * wy).view(B, -1).mean(dim=1)                         # (B,)
        per_sample_loss = smooth_x + smooth_y                               # (B,)
        return per_sample_loss.mean()


    time_train_start = time.time()
    step_start = sess.clock.step

    for epoch in range(sess.clock.epoch, sess.config.epoch_num):
        net.train()
        train_loader.sampler.set_epoch(epoch)  # 에폭별 셔플
        loss_record = 0.0
        depth_loss_record = 0.0
        confidence_loss_record = 0.0
        smooth_loss_record = 0.0
        mae_sum = 0.0
        rmse_sum = 0.0
        
        for idx, batch_data in enumerate(train_loader):
            images, radar, lidar, gt_confidence = batch_data
            images = images.to(device_id, non_blocking=True)
            radar  = radar.to(device_id,  non_blocking=True)
            lidar  = lidar.to(device_id,  non_blocking=True)
            gt_confidence = gt_confidence.to(device_id, non_blocking=True)
            
            
            logits, confidence_map, a, x = sess.net(images, radar)

            valid_mask = (lidar > 0).float()                           # (B,1,H,W)
            
            # 1) Depth loss (per-sample then batch mean)
            per_elem = huber_loss(logits, lidar)                       # (B,1,H,W)
            masked = per_elem * valid_mask
            counts = valid_mask.sum(dim=[1,2,3]).clamp(min=1.0)      # (B,)
            loss_per_sample = masked.sum(dim=[1,2,3]) / counts       # (B,)
            loss_depth = loss_per_sample.mean()                      # scalar

            # 2) Confidence loss (per-sample then batch mean)
            loss_confidence = bce_loss(confidence_map, gt_confidence).mean()

            # 3) Smoothness loss (already per-sample→batch mean)
            loss_smoothness = sobel_smoothness_loss(logits, images)
            
            
            loss_depth  = loss_depth * 1.0
            loss_confidence = loss_confidence * 10.0
            loss_smoothness = loss_smoothness * 0.1
            
            # 4) 최종 손실
            loss = loss_depth + loss_confidence + loss_smoothness

            # backward
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            lr_scheduler.step()
            
            with torch.no_grad():
                mask = (lidar > 0).float()
                counts = mask.sum(dim=[1,2,3]).clamp(min=1.0)
                mae_per_image = (torch.abs(logits - lidar) * mask).sum(dim=[1,2,3]) / counts
                rmse_per_image = torch.sqrt(((logits - lidar)**2 * mask).sum(dim=[1,2,3]) / counts)
            
            mae = mae_per_image.mean()
            rmse = rmse_per_image.mean()
            
            mae_sum += mae.item()
            rmse_sum += rmse.item()
            loss_record += loss.item()
            depth_loss_record += loss_depth.item()
            confidence_loss_record += loss_confidence.item()
            smooth_loss_record += loss_smoothness.item()

            # 로그
            sess.clock.tick()

            if RANK == 0 and (idx+1) % config.log_interval == 0:
                with torch.no_grad():
                    # (이미지 저장 및 시각화 관련 코드는 생략)
                    avg_loss = loss_record / config.log_interval
                    avg_depth_loss = depth_loss_record / config.log_interval
                    avg_confidence_loss = confidence_loss_record / config.log_interval
                    avg_smooth_loss = smooth_loss_record / config.log_interval
                    time_train_passed = time.time() - time_train_start
                    step_passed = sess.clock.step - step_start
                    eta = (total_step - sess.clock.step) / max(step_passed, 1e-9) * time_train_passed

                    confidence_map = torch.sigmoid(confidence_map)
                    
                    image = images[0].permute(1,2,0).detach().cpu().numpy()
                    logit = logits[0].detach().cpu().numpy()
                    confidence_map = confidence_map[0].detach().cpu().numpy()
                    gt_confidence = gt_confidence[0].detach().cpu().numpy()
                    lidar = lidar[0].detach().cpu().numpy()
                    radar = radar[0].detach().cpu().numpy()
                    radar = radar.repeat(image.shape[0], axis=1)
                    a = a[0].detach().cpu().numpy()
                    x = x[0].detach().cpu().numpy()
                    
                    plt.clf()
                    plt.axis('off')
                    plt.imshow(radar[0], cmap='inferno', vmin=0, vmax=80)
                    plt.savefig('./train_log/radar.png', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    plt.clf()
                    plt.axis('off')
                    image = (image - image.min()) / (image.max() - image.min())
                    plt.imshow(image)
                    plt.savefig('./train_log/img.png', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    plt.clf()
                    plt.axis('off')
                    plt.imshow(logit[0], cmap='inferno', vmin=0, vmax=80)
                    plt.savefig('./train_log/logit.png', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    plt.clf()
                    plt.axis('off')
                    plt.imshow(confidence_map[0], cmap='inferno')
                    plt.savefig('./train_log/quasi_conf.png', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    plt.clf()
                    plt.axis('off')
                    plt.imshow(gt_confidence[0] , cmap='inferno')
                    plt.savefig('./train_log/quasi_conf_gt.png', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    plt.clf()
                    plt.axis('off')
                    plt.imshow(expand_lidar_points(abs(logit[0] - lidar[0]) * (lidar[0] > 0), size=9), cmap='inferno')
                    plt.savefig('./train_log/diff.png', bbox_inches='tight', pad_inches=0)
                    plt.close()

                    plt.clf()
                    plt.axis('off')
                    plt.imshow(expand_lidar_points(lidar[0], size=9), cmap='inferno', vmin=0, vmax=80)
                    plt.savefig('./train_log/gt.png', bbox_inches='tight', pad_inches=0)
                    plt.close()

                    meta_info = [
                        f"epoch:{epoch}/{config.epoch_num}",
                        f"iter:{idx+1}/{len(train_loader)}",
                        f"loss:{avg_loss:.4f}",
                        f"depth:{avg_depth_loss:.4f}",
                        f"confidence:{avg_confidence_loss:.4f}",
                        f"smooth:{avg_smooth_loss:.4f}",
                        f"lr:{lr_scheduler.get_last_lr()[0]:.6f}",
                        f"passed:{format_time(time_train_passed)}",
                        f"eta:{format_time(eta)}",
                        f"MAE:{mae_sum / config.log_interval:.4f}",
                        f"RMSE:{rmse_sum / config.log_interval:.4f}"
                    ]
                    logger.info(", ".join(meta_info))
                    
                    loss_record = 0.0
                    depth_loss_record = 0.0
                    confidence_loss_record = 0.0
                    smooth_loss_record = 0.0
                    mae_sum = 0.0
                    rmse_sum = 0.0

        sess.clock.tock()

        if not args.local:
            torch.distributed.barrier()
        # epoch 종료 후 체크포인트
        if RANK == 0:
            if (sess.clock.epoch+1) % config.checkpoint_interval == 0:
                sess.save_checkpoint(f"epoch-{sess.clock.epoch}")
            sess.save_checkpoint('latest')
    
    logger.info("Training done.")
    sys.exit(0)


if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, exit.")
        os._exit(0)

