import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

import dataset


_worker_dataset = None
_worker_output_dir = None
_worker_dataset_name = None
_worker_overwrite = False


def _init_worker(depth_dataset, output_dir, dataset_name, overwrite):
    global _worker_dataset, _worker_output_dir, _worker_dataset_name, _worker_overwrite
    _worker_dataset = depth_dataset
    _worker_output_dir = Path(output_dir)
    _worker_dataset_name = dataset_name
    _worker_overwrite = overwrite


def _save_one(index):
    if _worker_dataset_name == 'zju':
        confidence_filename = _worker_dataset._sample_id(index) + '.npy'
    else:
        camera_filename = _worker_dataset.get_camera_filename(index)
        confidence_filename = camera_filename.split('/')[-1].replace('.jpg', '.npy')
    save_path = _worker_output_dir / confidence_filename

    if save_path.exists() and not _worker_overwrite:
        return False

    sample = _worker_dataset[index]
    confidence_map = sample[3].squeeze(0).astype(np.uint8)
    np.save(save_path, np.packbits(confidence_map, axis=-1))
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='nuscenes', choices=['nuscenes', 'zju'])
    parser.add_argument('--output-dir', type=str, default='confidence_map')

    parser.add_argument('--nuscenes-path', type=str, default='data/nuscenes_radar_5sweeps_infos_train.pkl')
    parser.add_argument('--nuscenes-root', type=str, default='data/nuscenes/samples')

    parser.add_argument('--zju-path', type=str, default='data/zju/train.txt')
    parser.add_argument('--zju-root', type=str, default='data/zju')

    parser.add_argument('--link-lidar', action='store_true')
    parser.add_argument('--rid-outliers', action='store_true')

    parser.add_argument(
        '--rule',
        type=str,
        default='dot',
        choices=['column', 'dot'],
        help='Confidence rule. Use column for nuScenes; ZJU supports column or dot.',
    )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--workers',
        type=int,
        default=min(8, os.cpu_count() or 1),
        help='Number of parallel save workers (0 or 1 disables multiprocessing)',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'zju':
        depth_dataset = dataset.ZJUDataset(
            path=args.zju_path,
            data_root=args.zju_root,
            rid_outliers=args.rid_outliers,
            augmentation=False,
            rotation=False,
            gpu_augmentation=False,
            confidence_path=None,
            confidence_rule=args.rule,
            generate_confidence=True,
            sort_by_timestamp=True,
            crop_to_input=False,
        )
    else:
        depth_dataset = dataset.NuScenesDataset(
            path=args.nuscenes_path,
            data_root=args.nuscenes_root,
            link_lidar=args.link_lidar,
            rid_outliers=args.rid_outliers,
            augmentation=False,
            rotation=False,
            gpu_augmentation=False,
            confidence_path=None,
            crop_to_input=False,
        )

    worker_count = max(0, args.workers)
    _init_worker(depth_dataset, output_dir, args.dataset, args.overwrite)
    indices = range(len(depth_dataset))

    if worker_count <= 1:
        for index in tqdm(indices, desc='saving confidence maps'):
            _save_one(index)
        return

    # fork shares the already-loaded dataset with workers via copy-on-write.
    # Fall back to the platform default on systems where fork is unavailable.
    methods = mp.get_all_start_methods()
    context = mp.get_context('fork' if 'fork' in methods else methods[0])
    with context.Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=(depth_dataset, str(output_dir), args.dataset, args.overwrite),
    ) as pool:
        for _ in tqdm(
            pool.imap_unordered(_save_one, indices, chunksize=1),
            total=len(depth_dataset),
            desc=f'saving confidence maps ({worker_count} workers)',
        ):
            pass


if __name__ == '__main__':
    main()
