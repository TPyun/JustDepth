import os
import cv2
import numpy as np
import pickle
from PIL import Image
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud

from utils import (
    map_pointcloud_to_image,
    get_lidar_map,
    get_radar_map,
    canvas_filter,
)

class NuScenesConf:
    input_h = 896
    input_w = 1600
    max_depth = 80
    min_depth = 0
    
    query_radius_outlier_x = 8
    query_radius_outlier_y = 16
    outlier_depth_threshold = 30
    
    link_distance = 48
    link_pass_x = 4
    link_pass_y = 16

class ZJUConf:
    input_h = 704
    input_w = 1280
    max_depth = 80
    min_depth = 0

    query_radius_outlier_x = 8
    query_radius_outlier_y = 16
    outlier_depth_threshold = 30

_rng_cache = {}


def _get_rng():
    """Return a reproducible RNG unique to each DataLoader worker and rank."""
    worker = torch.utils.data.get_worker_info()
    base_seed = worker.seed if worker is not None else torch.initial_seed()
    rank = int(os.environ.get('RANK', '0'))
    seed = (int(base_seed) + rank * 1_000_003) % (2 ** 63)
    key = (os.getpid(), seed)
    generator = _rng_cache.get(key)
    if generator is None:
        generator = np.random.default_rng(seed)
        _rng_cache[key] = generator
    return generator

# Backward-compatible alias: dataset.conf means nuScenes config.
conf = NuScenesConf

class _SharedInfoList:
    def __init__(self, items):
        blobs = [pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL) for x in items]
        self._buf = np.frombuffer(b"".join(blobs), dtype=np.uint8)
        off = np.zeros(len(items) + 1, dtype=np.int64)
        if blobs:
            off[1:] = np.cumsum([len(b) for b in blobs], dtype=np.int64)
        self._off = off

    def __len__(self):
        return len(self._off) - 1

    def __getitem__(self, i):
        s = int(self._off[i])
        e = int(self._off[i + 1])
        return pickle.loads(self._buf[s:e].tobytes())


def _safe_lidar_map(points, shape):
    if points is None or len(points) == 0:
        return np.zeros(shape, dtype=np.float32)
    return get_lidar_map(points, shape)


def _safe_radar_map(points, shape):
    if points is None or len(points) == 0:
        return np.zeros(shape, dtype=np.float32)
    return get_radar_map(points, shape)


def _crop_points_to_image(points, width, height):
    if points is None or len(points) == 0:
        return points
    keep = (
        (points[:, 0] >= 0) & (points[:, 0] < width) &
        (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    return points[keep]


def crop_sample_to_input(img, lidar, radar, confidence_map, input_h, input_w):
    width, height = img.size
    crop_top = max(height - input_h, 0)
    target_h = min(height, input_h)
    target_w = min(width, input_w)

    if crop_top or target_h != height or target_w != width:
        img = img.crop((0, crop_top, target_w, crop_top + target_h))

    def crop_points(points):
        if points is None or len(points) == 0:
            return points
        points = points.copy()
        points[:, 1] -= crop_top
        return _crop_points_to_image(points, target_w, target_h)

    lidar = crop_points(lidar)
    radar = crop_points(radar)

    if confidence_map is not None:
        if confidence_map.shape[0] > target_h:
            confidence_map = confidence_map[confidence_map.shape[0] - target_h:]
        confidence_map = confidence_map[:target_h, :target_w]

    return img, lidar, radar, confidence_map


class NuScenesDataset(torch.utils.data.Dataset):
    def __init__(self, 
        data_root = '.data/nuscenes/samples',
        path = './data/nuscenes_radar_5sweeps_infos_train.pkl',
        
        link_lidar=True,
        rid_outliers=True,
        
        augmentation=True,
        rotation=True,
        gpu_augmentation=False,
        rotation_deg=10.0,
        scene_hflip_probability=0.5,
        confidence_path=None,
        crop_to_input=True,
        generate_confidence=True,
        ):
        
        self.path = path
        self.data_root = data_root
        self.conf = NuScenesConf
        
        self.augmentation = augmentation
        self.rid_outliers = rid_outliers
        self.link_lidar = link_lidar
        self.rotation = rotation
        self.gpu_augmentation = bool(gpu_augmentation)
        self.rotation_deg = float(rotation_deg)
        self.scene_hflip_probability = float(scene_hflip_probability)
        self.confidence_path = confidence_path
        self.crop_to_input = bool(crop_to_input)
        self.generate_confidence = bool(generate_confidence)
        self._missing_confidence_warn_count = 0

        print('Loading data...')
        with open(self.path, 'rb') as f:
            self.infos = _SharedInfoList(pickle.loads(f.read()))
        print('Data loaded.')
        
        self.radar_use_type = 'RADAR_FRONT'
        self.camera_use_type = 'CAM_FRONT'
        self.lidar_use_type = 'LIDAR_TOP'
        
        print('Data length:', len(self.infos))
        self._print_confidence_cache_status()

    def __len__(self):
        return len(self.infos)
        
    def get_params(self, data):
        params = dict()
        if 'calibrated_sensor' in data.keys():
            params['sensor2ego'] = data['calibrated_sensor']
        else:
            params['sensor2ego'] = dict()
            params['sensor2ego']['translation'] = data['sensor2ego_translation']
            params['sensor2ego']['rotation'] = data['sensor2ego_rotation']
        
        if 'ego_pose' in data.keys():
            params['ego2global'] = data['ego_pose']
        else:
            params['ego2global'] = dict()
            params['ego2global']['translation'] = data['ego2global_translation']
            params['ego2global']['rotation'] = data['ego2global_rotation']
        
        return params
    
    def set_curr_epoch(self, epoch):
        self.curr_epoch = epoch

    def _print_confidence_cache_status(self):
        if self.confidence_path:
            if os.path.isdir(self.confidence_path):
                count = len([name for name in os.listdir(self.confidence_path) if name.endswith('.npy')])
                print(f'Using precomputed confidence maps: {self.confidence_path} ({count} files)')
            else:
                print(f'Precomputed confidence map path not found: {self.confidence_path}; generating maps on the fly.')
        elif not self.generate_confidence:
            print('Confidence map generation disabled; returning zero maps.')
        else:
            print('No precomputed confidence map path set; generating maps on the fly.')

    def get_camera_filename(self, index):
        data = self.infos[index]
        camera_infos = data['cam_infos'][self.camera_use_type]
        return camera_infos['filename'].split('samples/')[-1]

    def load_precomputed_confidence(self, camera_filename, image_width=None):
        if not self.confidence_path:
            return None
        confidence_filename = camera_filename.split('/')[-1].replace('.jpg', '.npy').replace('.png', '.npy')
        confidence_map_path = os.path.join(self.confidence_path, confidence_filename)
        if not os.path.exists(confidence_map_path):
            if self._missing_confidence_warn_count < 5:
                print(f'Missing precomputed confidence map: {confidence_map_path}; generating this sample on the fly.')
                self._missing_confidence_warn_count += 1
            return None
        confidence_map = np.load(confidence_map_path)
        target_width = self.conf.input_w if image_width is None else int(image_width)
        if confidence_map.ndim == 2 and confidence_map.shape[1] == target_width:
            decoded = confidence_map
        else:
            decoded = np.unpackbits(confidence_map, axis=-1)[:, :target_width]
        return np.asarray(decoded, dtype=np.uint8)

    def __getitem__(self, index):
        hflip = None
        forced_color = None
        forced_angle = None
        if isinstance(index, tuple):
            idx_tuple = index
            index = int(idx_tuple[0])
            if len(idx_tuple) > 1:
                hflip = idx_tuple[1]
            if len(idx_tuple) >= 6:
                forced_color = (float(idx_tuple[2]), float(idx_tuple[3]), float(idx_tuple[4]))
                forced_angle = float(idx_tuple[5])

        data = self.infos[index]
        
        # 카메라 이미지 로드
        camera_infos = data['cam_infos'][self.camera_use_type]
        camera_params = self.get_params(camera_infos)
        camera_filename = camera_infos['filename'].split('samples/')[-1]
        img_path = os.path.join(self.data_root, camera_filename)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'nuScenes image not found or unreadable: {img_path}')
        confidence_map = self.load_precomputed_confidence(camera_filename, image_width=img.shape[1])

        # 레이더 포인트클라우드 로드
        radar_infos = data['radar_infos'][self.radar_use_type][0]
        radar_params = self.get_params(radar_infos)
        radar_path = radar_infos['data_path'].split('samples/')[-1]
        radar_obj = RadarPointCloud.from_file(os.path.join(self.data_root, radar_path))
        radar_all = radar_obj.points.transpose(1,0)[:, :3]
        radar = np.concatenate((radar_all, np.ones([radar_all.shape[0], 1])), axis=1)

        # 라이다 포인트클라우드 로드
        lidar_infos = data['lidar_infos'][self.lidar_use_type]
        lidar_params = self.get_params(lidar_infos)
        lidar_path = lidar_infos['filename'].split('samples/')[-1]
        lidar_obj = LidarPointCloud.from_file(os.path.join(self.data_root, lidar_path))
        lidar_all = lidar_obj.points.transpose(1,0)[:, :3]
        lidar = np.concatenate((lidar_all, np.ones([lidar_all.shape[0], 1])), axis=1)

        # 포인트를 이미지 좌표로 투영
        _, lidar = map_pointcloud_to_image(lidar,
                                           lidar_params['sensor2ego'], lidar_params['ego2global'],
                                           camera_params['sensor2ego'], camera_params['ego2global'])
        _, radar = map_pointcloud_to_image(radar,
                                           radar_params['sensor2ego'], radar_params['ego2global'],
                                           camera_params['sensor2ego'], camera_params['ego2global'])

        inds = canvas_filter(lidar[:, :2], img.shape[:2])
        lidar = lidar[inds]
        inds = canvas_filter(radar[:, :2], img.shape[:2])
        radar = radar[inds]
        
        # 유효한 라이다 포인트 선택 및 깊이 제한 적용
        lidar = lidar[(lidar[:, 2] > self.conf.min_depth) & (lidar[:, 2] < self.conf.max_depth)]
        
        # 유효한 레이더 포인트 선택 및 깊이 제한 적용
        radar = radar[radar[:, 2] > self.conf.min_depth]

        if self.link_lidar:
            lidar = densify_lidar_points(lidar, pass_X = self.conf.link_pass_x, pass_Y=self.conf.link_pass_y, link_R=self.conf.link_distance, D=0.2)

        if self.rid_outliers:
            uvs, depths = lidar[:, :2], lidar[:, 2]
            tree_outlier = cKDTree(uvs)

            # 주변 이웃 찾기
            res_outlier = tree_outlier.query_ball_point(uvs, self.conf.query_radius_outlier_y)

            filter_mask = np.zeros(len(uvs), dtype=bool)
            for i, neighbors in enumerate(res_outlier):
                neighbors = [n for n in neighbors if np.abs(uvs[i][0] - uvs[n][0]) < self.conf.query_radius_outlier_x]
                if len(neighbors) < 2:
                    continue
                
                min_depth = np.min(depths[neighbors])
                if min_depth > self.conf.outlier_depth_threshold:
                    continue

                rel_diff = (depths[i] - min_depth) / depths[i]
                filter_mask[i] = (rel_diff > 0.1)
            lidar = lidar[~filter_mask]
            
        # PIL 이미지 변환
        img_pil = Image.fromarray(img[..., ::-1])  # BGR -> RGB
        
        aug_params = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        if self.augmentation and self.gpu_augmentation:
            width, height = img_pil.size
            sample_rng = _get_rng()
            hflip_b = bool(hflip) if hflip is not None else bool(sample_rng.uniform(0.0, 1.0) < self.scene_hflip_probability)
            if forced_color is not None:
                b_, c_, s_ = forced_color
            else:
                b_ = sample_rng.uniform(0.6, 1.4)
                c_ = sample_rng.uniform(0.6, 1.4)
                s_ = sample_rng.uniform(0.6, 1.4)
            if not self.rotation:
                angle = 0.0
            elif forced_angle is not None:
                angle = float(forced_angle)
            else:
                angle = float(sample_rng.uniform(-self.rotation_deg, self.rotation_deg))
            aug_params = np.array([0.0, float(b_), float(c_), float(s_), 0.0], dtype=np.float32)

            if confidence_map is None and self.generate_confidence:
                lid0 = np.asarray(get_lidar_map(lidar, (height, width)), dtype=np.float32)
                rad0 = np.asarray(get_radar_map(radar, (height, width)), dtype=np.float32)
                confidence_map = generate_confidence_map(
                    rad0, lid0, region_width=self.conf.input_w // 16, threshold=0.5
                )
            img_aug, lidar_aug, radar_aug, confidence_map = augment_geometry(
                img_pil, lidar, radar, confidence_map=confidence_map,
                angle=angle, hflip=hflip_b,
            )
        elif self.augmentation:
            if hflip is None:
                hflip = _get_rng().uniform(0.0, 1.0) < self.scene_hflip_probability
            img_aug, lidar_aug, radar_aug, confidence_map = augmention(
                img_pil, lidar, radar, confidence_map=confidence_map,
                rotation=self.rotation, hflip=bool(hflip), forced_angle=forced_angle,
            )
        else:
            img_aug, lidar_aug, radar_aug = img_pil, lidar, radar
        if self.crop_to_input:
            img_aug, lidar_aug, radar_aug, confidence_map = crop_sample_to_input(
                img_aug, lidar_aug, radar_aug, confidence_map, self.conf.input_h, self.conf.input_w
            )

        # lidar radar 맵 생성
        lidar_array = get_lidar_map(lidar_aug, (img_aug.height, img_aug.width))
        radar_array = get_radar_map(radar_aug, (img_aug.height, img_aug.width))
        
        # numpy 배열로 변환
        # GPU augmentation normalizes on the training device.  Keep images as
        # uint8 here to avoid a 4x larger worker-side copy and pinned transfer.
        img_np = np.asarray(img_aug)  # (H, W, C), uint8
        lidar_array = np.array(lidar_array, dtype=np.float32)
        radar_array = np.array(radar_array, dtype=np.float32)
        
        if confidence_map is None:
            if self.generate_confidence:
                confidence_map = generate_confidence_map(
                    radar_array, lidar_array, region_width=self.conf.input_w // 16, threshold=0.5
                )
            else:
                confidence_map = np.zeros((lidar_array.shape[0], lidar_array.shape[1]), dtype=np.float32)
        confidence_array = np.array(confidence_map, dtype=np.float32)
        
        # 차원 확장
        lidar_array = lidar_array[None]
        radar_array = radar_array[None]
        confidence_array = confidence_array[None]
        
        # 이미지 전처리
        img_np = np.ascontiguousarray(img_np.transpose(2, 0, 1))  # (C, H, W)
        if not self.gpu_augmentation:
            img_np = img_np.astype(np.float32, copy=False)
            img_np /= 255.0
            for c in range(3):
                img_np[c, :, :] = (img_np[c, :, :] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c]

        # 레이더를 (1, W) 형태로 변환
        radar_array = radar_array.squeeze()  # (H, W) -> (W,)
        mid_h = radar_array.shape[0] // 2
        radar_array = radar_array[mid_h].astype(np.float32)  # 중앙 값 선택
        radar_array = radar_array[None, None, :]  # (1, 1, W)

        if self.gpu_augmentation:
            return img_np, radar_array, lidar_array, confidence_array, aug_params
        return img_np, radar_array, lidar_array, confidence_array


def remove_lidar_outliers_fast(lidar, shape, query_radius_x=8, query_radius_y=16,
                               depth_threshold=30, rel_threshold=0.1):
    lidar = np.asarray(lidar, dtype=np.float32)
    if lidar.shape[0] == 0:
        return lidar
    H, W = int(shape[0]), int(shape[1])
    xs = np.clip(lidar[:, 0].astype(np.int64), 0, W - 1)
    ys = np.clip(lidar[:, 1].astype(np.int64), 0, H - 1)
    d = lidar[:, 2].astype(np.float32)

    dmap = np.zeros((H, W), np.float32)
    dmap[ys, xs] = d
    valid = (dmap > 0).astype(np.float32)
    big = np.where(dmap > 0, dmap, np.float32(1e9))

    ry = int(query_radius_y)
    rx = int(query_radius_x)
    dyy = np.arange(-ry, ry + 1)[:, None]
    dxx = np.arange(-(rx - 1), rx)[None, :]
    footprint = ((dxx * dxx + dyy * dyy) <= ry * ry).astype(np.uint8)

    local_min = cv2.erode(big, footprint, borderType=cv2.BORDER_CONSTANT, borderValue=1e9)
    count = cv2.filter2D(valid, -1, footprint.astype(np.float32), borderType=cv2.BORDER_CONSTANT)

    lm = local_min[ys, xs]
    cn = count[ys, xs]
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = (d - lm) / d
    remove = (cn >= 2) & (lm <= depth_threshold) & (rel > rel_threshold)
    return lidar[~remove]


class ZJUDataset(torch.utils.data.Dataset):
    def __init__(self,
        path='data/zju/train.txt',
        data_root='data/zju',
        rid_outliers=True,
        augmentation=True,
        rotation=False,
        gpu_augmentation=False,
        rotation_deg=10.0,
        scene_hflip_probability=0.5,
        confidence_path=None,
        confidence_rule='dot',
        depth_scale=256.0,
        generate_confidence=True,
        sort_by_timestamp=False,
        crop_to_input=True,
        ):

        self.path = path
        self.data_root = data_root
        self.rid_outliers = rid_outliers
        self.augmentation = augmentation
        self.rotation = rotation
        self.gpu_augmentation = bool(gpu_augmentation)
        self.rotation_deg = float(rotation_deg)
        self.scene_hflip_probability = float(scene_hflip_probability)
        self.confidence_path = confidence_path
        self.confidence_rule = str(confidence_rule)
        if self.confidence_rule == 'v1':
            self.confidence_rule = 'column'
        elif self.confidence_rule == 'v2':
            self.confidence_rule = 'dot'
        if self.confidence_rule not in ('column', 'dot'):
            raise ValueError("confidence_rule must be 'column' or 'dot'.")
        self.depth_scale = float(depth_scale)
        self.generate_confidence = bool(generate_confidence)
        self.crop_to_input = bool(crop_to_input)
        self.conf = ZJUConf
        self.input_h = self.conf.input_h
        self.input_w = self.conf.input_w
        self.max_depth = self.conf.max_depth
        self.min_depth = self.conf.min_depth
        self._missing_confidence_warn_count = 0

        print('Loading ZJU-4DRadarCam data...')
        with open(self.path, 'r') as f:
            self.infos = [line.strip() for line in f if line.strip()]
        if sort_by_timestamp:
            self.infos = sorted(self.infos, key=lambda x: os.path.splitext(os.path.basename(x))[0])
        print('ZJU data loaded.')
        print('Data length:', len(self.infos))
        self._print_confidence_cache_status()

    def __len__(self):
        return len(self.infos)

    def _sample_id(self, index):
        return os.path.splitext(os.path.basename(self.infos[index]))[0]

    def _print_confidence_cache_status(self):
        if self.confidence_path:
            if os.path.isdir(self.confidence_path):
                count = len([name for name in os.listdir(self.confidence_path) if name.endswith('.npy')])
                print(f'Using precomputed confidence maps: {self.confidence_path} ({count} files)')
            else:
                print(f'Precomputed confidence map path not found: {self.confidence_path}; generating maps on the fly.')
        elif not self.generate_confidence:
            print('Confidence map generation disabled; returning zero maps.')
        else:
            print('No precomputed confidence map path set; generating maps on the fly.')

    def load_precomputed_confidence(self, sample_id, image_width=None):
        if not self.confidence_path:
            return None
        confidence_map_path = os.path.join(self.confidence_path, sample_id + '.npy')
        if not os.path.exists(confidence_map_path):
            if self._missing_confidence_warn_count < 5:
                print(f'Missing precomputed confidence map: {confidence_map_path}; generating this sample on the fly.')
                self._missing_confidence_warn_count += 1
            return None
        confidence_map = np.load(confidence_map_path)
        target_width = self.input_w if image_width is None else int(image_width)
        if confidence_map.ndim == 2 and confidence_map.shape[1] == target_width:
            decoded = confidence_map
        else:
            decoded = np.unpackbits(confidence_map, axis=-1)[:, :target_width]
        return np.asarray(decoded, dtype=np.uint8)

    def __getitem__(self, index):
        hflip = None
        forced_color = None
        forced_angle = None
        if isinstance(index, tuple):
            idx_tuple = index
            index = int(idx_tuple[0])
            if len(idx_tuple) > 1:
                hflip = idx_tuple[1]
            if len(idx_tuple) >= 6:
                forced_color = (float(idx_tuple[2]), float(idx_tuple[3]), float(idx_tuple[4]))
                forced_angle = float(idx_tuple[5])

        sample_id = self._sample_id(index)
        img_path = os.path.join(self.data_root, 'image', sample_id + '.png')
        gt_path = os.path.join(self.data_root, 'gt', sample_id + '.png')
        radar_path = os.path.join(self.data_root, 'radar', sample_id + '.npy')

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'ZJU image not found or unreadable: {img_path}')
        confidence_map = self.load_precomputed_confidence(sample_id, image_width=img.shape[1])

        gt_depth = np.asarray(Image.open(gt_path), dtype=np.float32) / self.depth_scale
        gt_depth[(gt_depth <= self.min_depth) | (gt_depth >= self.max_depth)] = 0.0
        ys, xs = np.nonzero(gt_depth > 0)
        lidar = np.stack([xs.astype(np.float32), ys.astype(np.float32), gt_depth[ys, xs].astype(np.float32)], axis=1)

        radar = np.load(radar_path).astype(np.float32)
        if radar.ndim != 2 or radar.shape[1] < 3:
            radar = np.zeros((0, 3), dtype=np.float32)
        radar = radar[:, :3]

        inds = canvas_filter(lidar[:, :2], img.shape[:2])
        lidar = lidar[inds]
        inds = canvas_filter(radar[:, :2], img.shape[:2])
        radar = radar[inds]
        lidar = lidar[(lidar[:, 2] > self.min_depth) & (lidar[:, 2] < self.max_depth)]
        radar = radar[radar[:, 2] > self.min_depth]

        if self.rid_outliers and len(lidar) > 0:
            lidar = remove_lidar_outliers_fast(
                lidar, img.shape[:2],
                query_radius_x=self.conf.query_radius_outlier_x,
                query_radius_y=self.conf.query_radius_outlier_y,
                depth_threshold=self.conf.outlier_depth_threshold,
            )

        img_pil = Image.fromarray(img[..., ::-1])

        aug_params = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        if self.augmentation and self.gpu_augmentation:
            width, height = img_pil.size
            sample_rng = _get_rng()
            hflip_b = bool(hflip) if hflip is not None else bool(sample_rng.uniform(0.0, 1.0) < self.scene_hflip_probability)
            if forced_color is not None:
                b_, c_, s_ = forced_color
            else:
                b_ = sample_rng.uniform(0.6, 1.4)
                c_ = sample_rng.uniform(0.6, 1.4)
                s_ = sample_rng.uniform(0.6, 1.4)
            if not self.rotation:
                angle = 0.0
            elif forced_angle is not None:
                angle = float(forced_angle)
            else:
                angle = float(sample_rng.uniform(-self.rotation_deg, self.rotation_deg))
            aug_params = np.array([0.0, float(b_), float(c_), float(s_), 0.0], dtype=np.float32)

            if confidence_map is None and self.generate_confidence:
                lid0 = np.asarray(_safe_lidar_map(lidar, (height, width)), dtype=np.float32)
                if self.confidence_rule == 'dot':
                    confidence_map = generate_confidence_map_dot(
                        radar, lid0, region_width=self.input_w // 16, threshold=0.5
                    )
                else:
                    rad0 = np.asarray(_safe_radar_map(radar, (height, width)), dtype=np.float32)
                    confidence_map = generate_confidence_map(
                        rad0, lid0, region_width=self.input_w // 16, threshold=0.5
                    )
            img_aug, lidar_aug, radar_aug, confidence_map = augment_geometry(
                img_pil, lidar, radar, confidence_map=confidence_map,
                angle=angle, hflip=hflip_b,
            )
        elif self.augmentation:
            if hflip is None:
                hflip = _get_rng().uniform(0.0, 1.0) < self.scene_hflip_probability
            img_aug, lidar_aug, radar_aug, confidence_map = augmention(
                img_pil, lidar, radar, confidence_map=confidence_map,
                rotation=self.rotation, hflip=bool(hflip), forced_angle=forced_angle,
            )
        else:
            img_aug, lidar_aug, radar_aug = img_pil, lidar, radar
        if self.crop_to_input:
            img_aug, lidar_aug, radar_aug, confidence_map = crop_sample_to_input(
                img_aug, lidar_aug, radar_aug, confidence_map, self.input_h, self.input_w
            )

        lidar_array = _safe_lidar_map(lidar_aug, (img_aug.height, img_aug.width))
        radar_array = _safe_radar_map(radar_aug, (img_aug.height, img_aug.width))

        # GPU augmentation normalizes on the training device.  Keep images as
        # uint8 here to avoid a 4x larger worker-side copy and pinned transfer.
        img_np = np.asarray(img_aug)
        lidar_array = np.asarray(lidar_array, dtype=np.float32)
        radar_array = np.asarray(radar_array, dtype=np.float32)

        if confidence_map is None:
            if self.generate_confidence:
                if self.confidence_rule == 'dot':
                    confidence_map = generate_confidence_map_dot(
                        radar_aug, lidar_array, region_width=self.input_w // 16, threshold=0.5
                    )
                else:
                    confidence_map = generate_confidence_map(
                        radar_array, lidar_array, region_width=self.input_w // 16, threshold=0.5
                    )
            else:
                confidence_map = np.zeros((lidar_array.shape[0], lidar_array.shape[1]), dtype=np.float32)
        confidence_array = np.asarray(confidence_map, dtype=np.float32)

        lidar_array = lidar_array[None]
        radar_array = radar_array[None]
        confidence_array = confidence_array[None]

        img_np = np.ascontiguousarray(img_np.transpose(2, 0, 1))
        if not self.gpu_augmentation:
            img_np = img_np.astype(np.float32, copy=False)
            img_np /= 255.0
            for c in range(3):
                img_np[c, :, :] = (img_np[c, :, :] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c]

        radar_array = radar_array.squeeze()
        mid_h = radar_array.shape[0] // 2
        radar_array = radar_array[mid_h].astype(np.float32)
        radar_array = radar_array[None, None, :]

        if self.gpu_augmentation:
            return img_np, radar_array, lidar_array, confidence_array, aug_params
        return img_np, radar_array, lidar_array, confidence_array
    
def densify_lidar_points(lidar_points, pass_X, pass_Y, link_R, D):
    # x, y 좌표 추출
    xy = lidar_points[:, :2]
    z = lidar_points[:, 2]

    tree = cKDTree(xy)
    pairs = tree.query_pairs(r=link_R)
    # 평균 포인트를 저장할 리스트
    avg_points = []
    # 각 포인트 쌍에 대해 조건 검사 및 평균 포인트 계산
    for i, j in pairs:
        # y축으로 거리 가까우면 pass
        if np.abs(xy[i][1] - xy[j][1]) < pass_Y:
            continue
        # x축으로 거리 멀면 pass
        if np.abs(xy[i][0] - xy[j][0]) > pass_X:
            continue
        
        if abs(z[i] - z[j]) < D:
            avg = (lidar_points[i] + lidar_points[j]) / 2
            avg_points.append(avg)

    # 평균 포인트가 존재하면 원래 포인트에 추가
    if avg_points:
        return np.vstack((avg_points, lidar_points))
    else:
        return lidar_points
    
def generate_confidence_map(radar_array, lidar_array, region_width, threshold=0.5):
    """
    radar_lidar_map: numpy array of shape (H, W), radar 기반 confidence map
    lidar_array: numpy array of shape (1, H, W), 확장된 LiDAR 깊이 맵
    threshold: 거리 임계값
    Returns:
        binary_map: numpy array of shape (H, W), 이진 맵
    """
    height, width = lidar_array.shape[0], lidar_array.shape[1]
    mid_h = height // 2
    radar_array = radar_array[mid_h].astype(np.float32)  # 중앙 값 선택
    if radar_array.ndim > 1:
        radar_array = radar_array[..., 0]
        
    # lidar_array shape (1, H, W) -> (H, W)
    lidar_map = lidar_array.squeeze()

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    
    lidar_depth_values = lidar_map[grid_y, grid_x]
    lidar_xy = np.stack([grid_x, grid_y], axis=-1)
    
    non_zero_mask = lidar_depth_values > 0
    lidar_depth_values = lidar_depth_values[non_zero_mask]
    lidar_xy = lidar_xy[non_zero_mask]
    if len(lidar_depth_values) < 4:
        return np.zeros((height, width), dtype=np.float32)
    
    lidar_map = griddata(lidar_xy, lidar_depth_values, (grid_x, grid_y), method='linear')
    
    distance_threshold = height // 32
    distance_mask = np.zeros_like(lidar_map, dtype=np.uint8)
    for x, y in lidar_xy:
        distance_mask = cv2.circle(distance_mask, (x, y), distance_threshold, 1, -1)

    # 4. 임계값을 초과하는 영역을 NaN 처리
    lidar_map = np.where(distance_mask, lidar_map, np.nan)
        
    H = lidar_map.shape[0]
    W = lidar_map.shape[1]
    W_r = radar_array.shape[0]
    
    binary_map = np.zeros((H, W), dtype=np.float32)
    
    for i, r_val in enumerate(radar_array):
        if r_val == 0:
            continue
        center_x = i
        half_width = region_width // 2
        start_x = max(center_x - half_width, 0)
        end_x = min(center_x + half_width, W)
        # 해당 영역의 LiDAR 깊이 추출
        region_lidar = lidar_map[:, start_x:end_x]
        diff = np.abs(region_lidar - r_val)
        binary_region = (diff < threshold).astype(np.float32)
        # 누적: 한 영역이라도 1이면 최종 맵에서도 1
        binary_map[:, start_x:end_x] = np.maximum(binary_map[:, start_x:end_x], binary_region)
        
    # 0보다 큰 값은 1로 설정
    binary_map = (binary_map > 0).astype(np.float32)

    return binary_map


def generate_confidence_map_dot(radar_points, lidar_array, region_width, threshold=0.5, vertical_tol=None):
    """Dot-wise confidence GT for ZJU.

    column(generate_confidence_map)은 get_radar_map으로 한 컬럼의 radar depth를 하나로
    줄인 뒤 비교한다. dot은 4D radar point의 (x, y)를 중심으로 원형 영역을 만들고,
    그 안에서 point 각각의 depth를 LiDAR dense map과 비교한다.

    dot의 기본 원 지름은 21 px(반지름 10 px)이다. ``vertical_tol``이 주어지면 반지름을
    명시적으로 덮어쓴다. ``region_width``는 column 방식과의 호출 호환을 위해
    인자로 유지한다.
    """
    lidar_map = np.asarray(lidar_array).squeeze()
    height, width = lidar_map.shape[0], lidar_map.shape[1]

    binary_map = np.zeros((height, width), dtype=np.float32)
    if radar_points is None or len(radar_points) == 0:
        return binary_map

    ys, xs = np.nonzero(lidar_map > 0)
    if len(xs) < 4:
        return binary_map

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    lidar_xy = np.stack([xs, ys], axis=1)
    lidar_vals = lidar_map[ys, xs]
    dense = griddata(lidar_xy, lidar_vals, (grid_x, grid_y), method='linear')

    distance_threshold = height // 32
    distance_mask = np.zeros((height, width), dtype=np.uint8)
    for x, y in lidar_xy:
        cv2.circle(distance_mask, (int(x), int(y)), distance_threshold, 1, -1)
    dense = np.where(distance_mask, dense, np.nan)

    radius = max(1, int(vertical_tol) if vertical_tol is not None else 10)
    r = np.asarray(radar_points, dtype=np.float32)
    for p in r:
        rx = int(p[0])
        ry = int(p[1])
        rd = float(p[2])
        if rd <= 0 or rx < 0 or rx >= width or ry < 0 or ry >= height:
            continue

        x0 = max(rx - radius, 0)
        x1 = min(rx + radius + 1, width)
        y0 = max(ry - radius, 0)
        y1 = min(ry + radius + 1, height)

        local_y, local_x = np.ogrid[y0:y1, x0:x1]
        circle = (local_x - rx) ** 2 + (local_y - ry) ** 2 <= radius ** 2
        region = dense[y0:y1, x0:x1]
        match = circle & (np.abs(region - rd) < threshold)
        if match.any():
            sub = binary_map[y0:y1, x0:x1]
            np.maximum(sub, match.astype(np.float32), out=sub)

    return (binary_map > 0).astype(np.float32)


def rotate_with_reflect_padding(img: Image, angle: float):
    w, h = img.size
    # 1) pad 크기 계산 (가로/세로 중 큰 쪽의 20%)
    pad = int(np.ceil(max(w, h) * 0.2))
    
    # 2) reflect padding
    img_padded = TF.pad(img, padding=pad, padding_mode='reflect')

    # 4) 회전 (fill 없이도 reflect padding 덕분에 빈 공간이 자연스럽게 채워짐)
    rotated = TF.affine(
        img_padded,
        angle=angle,
        translate=(0, 0),
        scale=1.0,
        shear=0,
        interpolation=InterpolationMode.BICUBIC,
    )
    
    # 5) 원본 크기로 정확히 center crop
    rotated_cropped = TF.center_crop(rotated, (h, w))
    
    return rotated_cropped

def horizenal_flip_points(points, center_x):
    """
    points: numpy array of shape (N, 2) or (N, 3)
    center_x: x 좌표 기준선
    Returns:
        flipped_points: numpy array of shape (N, 2) or (N, 3)
    """
    flipped_points = points.copy()
    flipped_points[:, 0] = 2 * center_x - points[:, 0]
    return flipped_points

def rotate_points(points, angle, center_x, center_y):
    """
    points: numpy array of shape (N, 2) or (N, 3)
    angle: 회전 각도 (도 단위)
    center_x, center_y: 회전 중심 좌표
    Returns:
        rotated_points: numpy array of shape (N, 2) or (N, 3)
    """
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # 회전 행렬 적용
    rotated_points = points.copy()
    rotated_points[:, 0] -= center_x
    rotated_points[:, 1] -= center_y

    x_new = rotated_points[:, 0] * cos_angle - rotated_points[:, 1] * sin_angle
    y_new = rotated_points[:, 0] * sin_angle + rotated_points[:, 1] * cos_angle

    rotated_points[:, 0] = x_new + center_x
    rotated_points[:, 1] = y_new + center_y

    return rotated_points

def rotate_confidence_nearest_no_pad(confidence_map: np.ndarray, angle: float) -> np.ndarray:
    if confidence_map.dtype != np.float32:
        confidence_map = confidence_map.astype(np.float32, copy=False)
    conf_pil = Image.fromarray(confidence_map, mode='F')
    rotated = TF.affine(
        conf_pil,
        angle=angle,
        translate=(0, 0),
        scale=1.0,
        shear=0,
        interpolation=InterpolationMode.NEAREST,
        fill=0,
    )
    return np.array(rotated, dtype=np.float32)

def augment_geometry(img: Image, lidar, radar, confidence_map=None, angle=0.0, hflip=False):
    width, height = img.size
    
    if hflip:
        img   = TF.hflip(img)
        center_x = (width - 1) / 2.0
        lidar = horizenal_flip_points(lidar, center_x)
        radar = horizenal_flip_points(radar, center_x)
        if confidence_map is not None:
            confidence_map = np.fliplr(confidence_map).copy()

    if angle != 0.0:
        img = rotate_with_reflect_padding(img, angle)
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        lidar = rotate_points(lidar, angle, center_x, center_y)
        radar = rotate_points(radar, angle, center_x, center_y)
        if confidence_map is not None:
            if confidence_map.shape[:2] == (height, width):
                confidence_map = rotate_confidence_nearest_no_pad(confidence_map, angle)
            else:
                confidence_map = None

        lidar = _crop_points_to_image(lidar, width, height)
        radar = _crop_points_to_image(radar, width, height)

    return img, lidar, radar, confidence_map


def augmention(img: Image, lidar, radar, confidence_map=None, rotation=False, hflip=False, forced_angle=None):
    sample_rng = _get_rng()
    if not rotation:
        angle = 0.0
    elif forced_angle is not None:
        angle = float(forced_angle)
    else:
        angle = float(sample_rng.uniform(-10.0, 10.0))
    img, lidar, radar, confidence_map = augment_geometry(
        img, lidar, radar, confidence_map=confidence_map, angle=angle, hflip=hflip
    )

    # Color jitter
    brightness = sample_rng.uniform(0.6, 1.4)
    contrast   = sample_rng.uniform(0.6, 1.4)
    saturation = sample_rng.uniform(0.6, 1.4)
            
    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)
    img = TF.adjust_saturation(img, saturation)

    return img, lidar, radar, confidence_map
