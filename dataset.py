import os
import cv2
import time
import numpy as np
import pickle
from PIL import Image, ImageOps
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from sklearn.cluster import DBSCAN
from collections import defaultdict

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision import transforms

from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud

from utils import (
    map_pointcloud_to_image,
    get_lidar_map,
    get_radar_map,
    canvas_filter,
    expand_lidar_points,
)

class conf:
    input_h = 896
    input_w = 1600
    max_depth = 100
    min_depth = 0
    
    query_radius_outlier_x = 8
    query_radius_outlier_y = 16
    outlier_depth_threshold = 30
    
    link_distance = 48
    link_pass_x = 4
    link_pass_y = 16

rng = np.random.default_rng()

class RCDepthDataset(torch.utils.data.Dataset):
    def __init__(self, 
        data_root = '.data/nuscenes/samples',
        path = './data/nuscenes_radar_5sweeps_infos_train.pkl',
        
        link_lidar=True,
        rid_outliers=True,
        
        augmentation=True,
        rotation=True,
        ):
        
        self.path = path
        self.data_root = data_root
        
        self.augmentation = augmentation
        self.rid_outliers = rid_outliers
        self.link_lidar = link_lidar
        self.rotation = rotation

        print('Loading data...')
        with open(self.path, 'rb') as f:
            self.infos = pickle.loads(f.read())
        print('Data loaded.')
        
        self.radar_use_type = 'RADAR_FRONT'
        self.camera_use_type = 'CAM_FRONT'
        self.lidar_use_type = 'LIDAR_TOP'
        
        print('Data length:', len(self.infos))

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

    def __getitem__(self, index):
        data = self.infos[index]
        
        # 카메라 이미지 로드
        camera_infos = data['cam_infos'][self.camera_use_type]
        camera_params = self.get_params(camera_infos)
        camera_filename = camera_infos['filename'].split('samples/')[-1]
        img_path = os.path.join(self.data_root, camera_filename)
        img = cv2.imread(img_path)

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
        lidar = lidar[(lidar[:, 2] > conf.min_depth) & (lidar[:, 2] < conf.max_depth)]
        
        # 유효한 레이더 포인트 선택 및 깊이 제한 적용
        radar = radar[radar[:, 2] > conf.min_depth]

        if self.link_lidar:
            lidar = densify_lidar_points(lidar, pass_X = conf.link_pass_x, pass_Y=conf.link_pass_y, link_R=conf.link_distance, D=0.2)

        if self.rid_outliers:
            uvs, depths = lidar[:, :2], lidar[:, 2]
            tree_outlier = cKDTree(uvs)

            # 주변 이웃 찾기
            res_outlier = tree_outlier.query_ball_point(uvs, conf.query_radius_outlier_y)

            filter_mask = np.zeros(len(uvs), dtype=bool)
            for i, neighbors in enumerate(res_outlier):
                neighbors = [n for n in neighbors if np.abs(uvs[i][0] - uvs[n][0]) < conf.query_radius_outlier_x]
                if len(neighbors) < 2:
                    continue
                
                min_depth = np.min(depths[neighbors])
                if min_depth > conf.outlier_depth_threshold:
                    continue

                rel_diff = (depths[i] - min_depth) / depths[i]
                filter_mask[i] = (rel_diff > 0.1)
            lidar = lidar[~filter_mask]
            
        # PIL 이미지 변환
        img_pil = Image.fromarray(img[..., ::-1])  # BGR -> RGB
        
        # 데이터 증강
        if self.augmentation:
            img_aug, lidar_aug, radar_aug = augmention(
                img_pil, lidar, radar, rotation=self.rotation
            )
        else:
            img_aug, lidar_aug, radar_aug = img_pil, lidar, radar
        # lidar radar 맵 생성
        lidar_array = get_lidar_map(lidar_aug, (img_aug.height, img_aug.width))
        radar_array = get_radar_map(radar_aug, (img_aug.height, img_aug.width))
        
        # numpy 배열로 변환
        img_np = np.array(img_aug, dtype=np.float32)  # (H, W, C)
        lidar_array = np.array(lidar_array, dtype=np.float32)
        radar_array = np.array(radar_array, dtype=np.float32)
        
        # 이미지 사이즈가 conf와 다르면 위쪽은 제거
        if img_np.shape[0] > conf.input_h:
            rid_height = img_np.shape[0] - conf.input_h
            img_np = img_np[rid_height:]
        if lidar_array.shape[0] > conf.input_h:
            rid_height = lidar_array.shape[0] - conf.input_h
            lidar_array = lidar_array[rid_height:]
        if radar_array.shape[0] > conf.input_h:
            rid_height = radar_array.shape[0] - conf.input_h
            radar_array = radar_array[rid_height:]
        
        confidence_map = generate_confidence_map(radar_array, lidar_array, region_width=conf.input_w // 16, threshold=0.5)
        confidence_array = np.array(confidence_map, dtype=np.float32)
        
        # 차원 확장
        lidar_array = lidar_array[None]
        radar_array = radar_array[None]
        confidence_array = confidence_array[None]
        
        # 이미지 전처리
        img_np = np.ascontiguousarray(img_np.transpose(2, 0, 1))  # (C, H, W)
        
        # 0 ~ 255 -> 0 ~ 1 범위로
        img_np /= 255.0

        # 채널별 평균과 표준편차로 정규화
        for c in range(3):
            img_np[c, :, :] = (img_np[c, :, :] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c]

        # 레이더를 (1, W) 형태로 변환
        radar_array = radar_array.squeeze()  # (H, W) -> (W,)
        mid_h = radar_array.shape[0] // 2
        radar_array = radar_array[mid_h].astype(np.float32)  # 중앙 값 선택
        radar_array = radar_array[None, None, :]  # (1, 1, W)

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
        
    # lidar_array shape (1, H, W) -> (H, W)
    lidar_map = lidar_array.squeeze()

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    
    lidar_depth_values = lidar_map[grid_y, grid_x]
    lidar_xy = np.stack([grid_x, grid_y], axis=-1)
    
    non_zero_mask = lidar_depth_values > 0
    lidar_depth_values = lidar_depth_values[non_zero_mask]
    lidar_xy = lidar_xy[non_zero_mask]
    
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

def augmention(img: Image, lidar, radar, rotation=False):
    width, height = img.size
    
    # 수평 뒤집기
    h_flip_p = 0.5
    if rng.uniform(0.0, 1.0) < h_flip_p:
        img   = TF.hflip(img)
        lidar = horizenal_flip_points(lidar, width / 2)
        radar = horizenal_flip_points(radar, width / 2)

    # Color jitter
    brightness = rng.uniform(0.6, 1.4)
    contrast   = rng.uniform(0.6, 1.4)
    saturation = rng.uniform(0.6, 1.4)
            
    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)
    img = TF.adjust_saturation(img, saturation)

    if rotation:
        angle = rng.uniform(-10.0, 10.0)

        # 어파인 변환 (모든 modality에 동일하게 적용)
        img = rotate_with_reflect_padding(img, angle)
        lidar = rotate_points(lidar, angle, width / 2, height / 2)
        radar = rotate_points(radar, angle, width / 2, height / 2)
        
        # 변환 후 이미지 크기가 원래보다 큰 경우 중앙 크롭 적용
        if width > width or height > height:
            img = TF.center_crop(img, (height, width))
            lidar = canvas_filter(lidar, (height, width))
            radar = canvas_filter(radar, (height, width))

    return img, lidar, radar
