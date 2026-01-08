import functools
import time

import scipy
import numpy as np
import torch

from nuscenes.utils.data_classes import Box, LidarPointCloud, RadarPointCloud
from pyquaternion import Quaternion
from matplotlib import cm


def project_3d_to_2d(points: np.ndarray, projection_matrix: np.ndarray):
    """From vod.frame without rounding to int"""

    uvw = projection_matrix.dot(points.T)
    uvw /= uvw[2]
    uvs = uvw[:2].T
    # uvs = np.round(uvs).astype(np.int)

    return uvs


def map_pointcloud1_to_pointcloud2(
    lidar_points,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):
    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.
    
    lidar_points = LidarPointCloud(lidar_points.T)
    lidar_points.rotate(
        Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    lidar_points.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_ego_pose['translation']))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    lidar_points.translate(-np.array(cam_ego_pose['translation']))
    lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
    lidar_points.rotate(
        Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)
    
    points = lidar_points.points.transpose((1, 0))
    return points


def map_pointcloud_to_image(
    lidar_points,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):
    points = map_pointcloud1_to_pointcloud2(lidar_points, lidar_calibrated_sensor, lidar_ego_pose,
                                            cam_calibrated_sensor, cam_ego_pose, min_dist)

    uvs = project_3d_to_2d(points[:, :3], np.array(cam_calibrated_sensor['camera_intrinsic']))

    return points, np.concatenate((uvs, points[:, 2:3]), 1)


def canvas_filter(data, shape):
    return np.all((data > 0) & (data < shape[1::-1]), 1)


def _scale_pts(data, out_shape, input_shape):
    data[:, :2] *= (np.array(out_shape[::-1]) / input_shape[1::-1])
    return data


def get_lidar_map(data, shape, input_shape=None):
    if input_shape is not None:
        data = _scale_pts(data.copy(), shape, input_shape)

    if np.any(data[:, :2].max(0) >= shape[1::-1]) or data[:, :2].min() < 0:
        inds = canvas_filter(data[:, :2], shape)
        data = data[inds]
    
    depth = np.zeros(shape + (data.shape[1] - 2, ), dtype=np.float32)
    depth[data[:, 1].astype(int), data[:, 0].astype(int)] = data[:, 2:]
    return depth.squeeze()


def get_radar_map(data, shape, input_shape=None):
    if input_shape is not None:
        data = _scale_pts(data.copy(), shape, input_shape)
        
    data = data[np.argsort(data[:, 2])[::-1]]

    depth = np.zeros(shape + (data.shape[1] - 2, ), dtype=np.float32)
    if np.any(data[:, :2].max(0) >= shape[1::-1]) or data[:, :2].min() < 0:
        inds = canvas_filter(data[:, :2], shape)
        data = data[inds]
    depth[:, data[:, 0].astype(int)] = data[:, 2:]
    return depth.squeeze()


def expand_lidar_points(lidar_img, size=3):
    H, W = lidar_img.shape
    radius = size // 2
    
    expanded = np.zeros_like(lidar_img, dtype=np.float32)

    ys, xs = np.where(lidar_img > 0)
    sorted_inds = np.argsort(lidar_img[ys, xs])[::-1]
    ys, xs = ys[sorted_inds], xs[sorted_inds]
    
    for y, x in zip(ys, xs):
        val = lidar_img[y, x]
        
        y0 = int(max(0, y - radius))
        y1 = int(min(H, y + radius + 1))
        x0 = int(max(0, x - radius))
        x1 = int(min(W, x + radius + 1))

        expanded[y0:y1, x0:x1] = np.where(expanded[y0:y1, x0:x1] == 0, val, expanded[y0:y1, x0:x1])
    
    return expanded
