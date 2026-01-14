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


logger.remove()

import timm
from timm.models import create_model
import model

from thop import profile
import dataset

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='blitzdepth', type=str, help='timm 모델 이름')
    parser.add_argument('--checkpoint', default=None, type=str, help='평가할 체크포인트 경로')
    args = parser.parse_args()

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

    dummy_img = torch.randn(1, 3, dataset.conf.input_h, dataset.conf.input_w).to(device)
    dummy_radar = torch.randn(1, 1, 1, 1600).to(device)
    net.eval()
    
    flops, params = profile(net, inputs=(dummy_img, dummy_radar))
    print(f'flops: {format_num(flops)}, params: {format_num(params)}')

    eval_dataset = dataset.RCDepthDataset(
        path='./data/nuscenes_radar_5sweeps_infos_test.pkl',
        augmentation=False,
        rid_outliers=False,
        link_lidar=False
    )
    eval_loader = DataLoader(eval_dataset, **config.params)
    logger.info(f"Eval dataset size = {len(eval_dataset)}")

    results_dir = Path(config.log_dir) / "vis_results"
    ensure_dir(results_dir)

    ranges = [50, 70, 80]
    
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
            img, radar, lidar, gt_confidence = batch_data
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
                valid_mask = (lidar_b > 0)
                if valid_mask.sum() > 0:
                    mae_val = np.abs(pred_b - lidar_b)[valid_mask].mean()
                    rmse_val = np.sqrt(np.square(pred_b - lidar_b)[valid_mask].mean())
                    mae_sum_total  += mae_val
                    rmse_sum_total += rmse_val
                    count_total    += 1

                for r in ranges:
                    pred_clamped = np.clip(pred_b, dataset.conf.min_depth, r)
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