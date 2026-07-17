import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
import cv2


logger.remove()

import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
import model

from thop import profile
import dataset
from run_config import (
    load_run_config,
    cfg_get,
    cfg_get_with_cli_priority,
    cfg_get_bool,
    cfg_get_int,
    cfg_get_list,
)

class config:
    log_dir = './eval_log'
    # DataLoader 설정
    params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 1,
        'persistent_workers': True,
        'prefetch_factor': 2,
    }

def ensure_dir(path: Path):
    path = Path(path)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)

# 단위 변환 함수: 1e3 -> K, 1e6 -> M, 1e9 -> B
def format_num(num):
    if num >= 1e9:
        return f"{num / 1e9:.2f} G"
    elif num >= 1e6:
        return f"{num / 1e6:.2f} M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f} K"
    else:
        return f"{num}"


def image_tensor_to_uint8(image):
    image = image.detach().cpu().float().numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)

    if image.min() < 0.0 or image.max() <= 4.0:
        mean = np.asarray(IMAGENET_DEFAULT_MEAN, dtype=np.float32)
        std = np.asarray(IMAGENET_DEFAULT_STD, dtype=np.float32)
        image = image * std + mean

    if image.max() <= 1.5:
        image = image * 255.0

    return np.clip(image, 0, 255).astype(np.uint8)


def depth_to_colormap(depth, min_depth, max_depth):
    depth = np.asarray(depth, dtype=np.float32)
    valid = depth > min_depth
    depth_vis = np.clip(depth, min_depth, max_depth)
    depth_vis = (depth_vis - min_depth) / max(max_depth - min_depth, 1e-6)
    depth_vis = (depth_vis * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color


def save_input_depth_pair(image, depth, save_path, min_depth, max_depth):
    image_vis = image_tensor_to_uint8(image)
    depth_vis = depth_to_colormap(depth, min_depth, max_depth)
    pair = np.concatenate([image_vis, depth_vis], axis=1)
    cv2.imwrite(str(save_path), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/nuscenes_eval.txt', type=str, help='Run config txt file')
    parser.add_argument('--model-name', default='blitzdepth', type=str, help='timm 모델 이름')
    parser.add_argument('--checkpoint', default=None, type=str, help='평가할 체크포인트 경로')
    parser.add_argument('--dataset', default='nuscenes', choices=['nuscenes', 'zju'], help='Eval dataset')
    args = parser.parse_args()
    run_cfg = load_run_config(args.config)

    args.model_name = cfg_get_with_cli_priority(run_cfg, 'model_name', args.model_name, '--model-name')
    args.checkpoint = cfg_get_with_cli_priority(run_cfg, 'checkpoint', args.checkpoint, '--checkpoint')
    args.dataset = cfg_get_with_cli_priority(run_cfg, 'dataset', args.dataset, '--dataset')
    config.log_dir = cfg_get(run_cfg, 'log_dir', config.log_dir)
    config.params.update({
        'batch_size': cfg_get_int(run_cfg, 'batch_size', config.params['batch_size']),
        'shuffle': cfg_get_bool(run_cfg, 'shuffle', config.params['shuffle']),
        'num_workers': cfg_get_int(run_cfg, 'num_workers', config.params['num_workers']),
        'persistent_workers': cfg_get_bool(run_cfg, 'persistent_workers', config.params['persistent_workers']),
        'prefetch_factor': cfg_get_int(run_cfg, 'prefetch_factor', config.params['prefetch_factor']),
    })
    save_visuals = cfg_get_bool(run_cfg, 'save_visuals', True)
    vis_interval = max(cfg_get_int(run_cfg, 'vis_interval', 1), 1)
    vis_max = cfg_get_int(run_cfg, 'vis_max', 0)
    ds_conf = dataset.ZJUConf if args.dataset == 'zju' else dataset.NuScenesConf

    # 로그 및 결과 디렉토리 설정
    ensure_dir(config.log_dir)
    log_path = Path(config.log_dir) / "eval_log.log"
    logger.add(str(log_path.resolve()), colorize=True, level="INFO",
               format="<green>[{time:%m-%d %H:%M:%S}]</green> {message}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 생성 및 체크포인트 로드
    net = create_model(
        model_name=args.model_name,
        pretrained=False,
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckp = torch.load(args.checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckp['network'], strict=False)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")

    dummy_img = torch.randn(1, 3, ds_conf.input_h, ds_conf.input_w).to(device)
    dummy_radar = torch.randn(1, 1, 1, ds_conf.input_w).to(device)
    net.eval()
    
    flops, params = profile(net, inputs=(dummy_img, dummy_radar))
    print(f'flops: {format_num(flops)}, params: {format_num(params)}')

    if args.dataset == 'zju':
        eval_dataset = dataset.ZJUDataset(
            path=cfg_get(run_cfg, 'dataset_path', 'data/zju/test.txt'),
            data_root=cfg_get(run_cfg, 'data_root', 'data/zju'),
            rid_outliers=cfg_get_bool(run_cfg, 'rid_outliers', False),
            augmentation=cfg_get_bool(run_cfg, 'augmentation', False),
            generate_confidence=cfg_get_bool(run_cfg, 'generate_confidence', False),
            sort_by_timestamp=cfg_get_bool(run_cfg, 'sort_by_timestamp', True),
        )
    else:
        eval_dataset = dataset.NuScenesDataset(
            path=cfg_get(run_cfg, 'dataset_path', './data/nuscenes_radar_5sweeps_infos_test.pkl'),
            data_root=cfg_get(run_cfg, 'data_root', '.data/nuscenes/samples'),
            augmentation=cfg_get_bool(run_cfg, 'augmentation', False),
            rid_outliers=cfg_get_bool(run_cfg, 'rid_outliers', False),
            link_lidar=cfg_get_bool(run_cfg, 'link_lidar', False),
            generate_confidence=cfg_get_bool(run_cfg, 'generate_confidence', False),
        )
    eval_loader = DataLoader(eval_dataset, **config.params)
    logger.info(f"Eval dataset size = {len(eval_dataset)}")

    results_dir = Path(config.log_dir) / "vis_results"
    if save_visuals:
        ensure_dir(results_dir)

    ranges = cfg_get_list(run_cfg, 'ranges', [50, 70, 80], int)
    
    mae_sum_total = 0.0
    rmse_sum_total = 0.0
    count_total    = 0
    total_forward_time = 0.0
    sample_counter = 0

    # Initialize metrics dictionary for range-specific metrics if needed.
    metrics_dict = {}
    for r in ranges:
        metrics_dict[r] = {'mae_sum': 0.0, 'rmse_sum': 0.0, 'count': 0}

    # Create a tqdm progress bar over the evaluation loader
    progress_bar = tqdm(eval_loader, desc="Evaluating")
    
    with torch.no_grad():
        for idx, batch_data in enumerate(progress_bar):
            img, radar, lidar, gt_confidence = batch_data[:4]
            img = img.to(device)
            radar = radar.to(device)
            lidar = lidar.to(device)
            B = img.size(0)
            sample_counter += B

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

            logits = net(img, radar, get_confidence=False)

            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            total_forward_time += elapsed_time_ms

            logits_np = logits.cpu().numpy()
            lidar_np  = lidar.cpu().numpy()
            
            for b in range(B):
                pred_b  = logits_np[b, 0]
                lidar_b = lidar_np[b, 0]
                sample_index = sample_counter - B + b
                should_save_vis = (
                    save_visuals
                    and sample_index % vis_interval == 0
                    and (vis_max <= 0 or sample_index < vis_max)
                )
                if should_save_vis:
                    save_input_depth_pair(
                        img[b],
                        pred_b,
                        results_dir / f"{sample_index:06d}.png",
                        ds_conf.min_depth,
                        ds_conf.max_depth,
                    )
                valid_mask = (lidar_b > 0)
                if valid_mask.sum() > 0:
                    mae_val = np.abs(pred_b - lidar_b)[valid_mask].mean()
                    rmse_val = np.sqrt(np.square(pred_b - lidar_b)[valid_mask].mean())
                    mae_sum_total  += mae_val
                    rmse_sum_total += rmse_val
                    count_total    += 1

                for r in ranges:
                    pred_clamped = np.clip(pred_b, ds_conf.min_depth, r)
                    mask_r = (lidar_b > 0) & (lidar_b <= r)
                    if mask_r.sum() > 0:
                        mae_r = np.abs(pred_clamped - lidar_b)[mask_r].mean()
                        rmse_r = np.sqrt(np.square(pred_clamped - lidar_b)[mask_r].mean())
                        metrics_dict[r]['mae_sum']  += mae_r
                        metrics_dict[r]['rmse_sum'] += rmse_r
                        metrics_dict[r]['count']    += 1

            # Calculate current average values
            if count_total > 0:
                avg_mae = mae_sum_total / count_total
                avg_rmse = rmse_sum_total / count_total
            else:
                avg_mae, avg_rmse = 0.0, 0.0

            # Update tqdm progress bar with current averages
            progress_bar.set_postfix({
                'MAE': f"{avg_mae:.4f}",
                'RMSE': f"{avg_rmse:.4f}",
            })
            

    if count_total > 0:
        mae_total = mae_sum_total / count_total
        rmse_total = rmse_sum_total / count_total


        logger.info(f"[All Range] count={count_total}, MAE={mae_total:.4f}, RMSE={rmse_total:.4f}")
        print(f"[All Range] count={count_total}, MAE={mae_total:.4f}, RMSE={rmse_total:.4f}")
    else:
        logger.info("[All Range] No valid pixels found!")

    for r in ranges:
        c = metrics_dict[r]['count']
        if c > 0:
            mae_r = metrics_dict[r]['mae_sum'] / c
            rmse_r = metrics_dict[r]['rmse_sum'] / c
            logger.info(f"[0~{r}m] count={c}, MAE={mae_r:.4f}, RMSE={rmse_r:.4f}")
            print(f"[0~{r}m] count={c}, MAE={mae_r:.4f}, RMSE={rmse_r:.4f}")
        else:
            logger.info(f"[0~{r}m] No valid pixels found")
    
    if sample_counter > 0:
        avg_forward_time_per_sample = total_forward_time / sample_counter
        logger.info(f"Total forward time={total_forward_time/1000:.4f} sec, "
                    f"Avg forward time per sample={avg_forward_time_per_sample:.3f} ms")
        print(f"Total forward time={total_forward_time/1000:.4f} sec, "
              f"Avg forward time per sample={avg_forward_time_per_sample:.3f} ms")
    else:
        logger.info("No sample processed, can't measure forward time")

    logger.info("Evaluation finished.")
    sys.exit(0)

if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, exit.")
        os._exit(0)
