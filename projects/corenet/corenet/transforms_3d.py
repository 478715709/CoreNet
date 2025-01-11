# modify from https://github.com/mit-han-lab/bevfusion
from typing import Any, Dict

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from PIL import Image
import mmcv
import random
from mmdet3d.datasets import GlobalRotScaleTrans
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.ops import box_np_ops as box_np_ops
from mmdet3d.structures.points import BasePoints
from .camera_corruptions import *
from .lidar_corruptions import *
@TRANSFORMS.register_module()
class ImageAug3D(BaseTransform):

    def __init__(self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
                 is_train):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate, mode):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode = mode)
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data['img']
        new_imgs = []
        transforms = []
        for i in range(len(imgs)):
            img = imgs[i]
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
                mode = 'RGB'
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform.numpy())
        data['img'] = new_imgs
        # update the calibration matrices
        data['img_aug_matrix'] = transforms
        return data


@TRANSFORMS.register_module()
class BEVFusionRandomFlip3D:
    """Compared with `RandomFlip3D`, this class directly records the lidar
    augmentation matrix in the `data`."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('horizontal')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('horizontal')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, :, ::-1].copy()

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('vertical')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('vertical')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, ::-1, :].copy()

        if 'lidar_aug_matrix' not in data:
            data['lidar_aug_matrix'] = np.eye(4)
        data['lidar_aug_matrix'][:3, :] = rotation @ data[
            'lidar_aug_matrix'][:3, :]
        return data


@TRANSFORMS.register_module()
class BEVFusionGlobalRotScaleTrans(GlobalRotScaleTrans):
    """Compared with `GlobalRotScaleTrans`, the augmentation order in this
    class is rotation, translation and scaling (RTS)."""

    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._trans_bbox_points(input_dict)
        self._scale_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'T', 'S'])

        lidar_augs = np.eye(4)
        lidar_augs[:3, :3] = input_dict['pcd_rotation'].T * input_dict[
            'pcd_scale_factor']
        lidar_augs[:3, 3] = input_dict['pcd_trans'] * \
            input_dict['pcd_scale_factor']

        if 'lidar_aug_matrix' not in input_dict:
            input_dict['lidar_aug_matrix'] = np.eye(4)
        input_dict[
            'lidar_aug_matrix'] = lidar_augs @ input_dict['lidar_aug_matrix']

        return input_dict


@TRANSFORMS.register_module()
class GridMask(BaseTransform):

    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def transform(self, results):
        if np.random.rand() > self.prob:
            return results
        imgs = results['img']
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.length = np.random.randint(1, d)
        else:
            self.length = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.length, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.length, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h,
                    (ww - w) // 2:(ww - w) // 2 + w]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(imgs[0])
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        results.update(img=imgs)
        return results

@TRANSFORMS.register_module()
class VelocityAug(object): # copy from DAL
    def __init__(self, rate=0.5, rate_vy=0.2, rate_rotation=-1, speed_range=None, thred_vy_by_vx=1.0,
                 ego_cam='CAM_FRONT'):
        # must be identical to that in tools/create_data_bevdet.py
        self.cls = ['car', 'truck', 'construction_vehicle',
                    'bus', 'trailer', 'barrier',
                    'motorcycle', 'bicycle',
                    'pedestrian', 'traffic_cone']
        self.speed_range = dict(
            car=[-10, 30, 6],
            truck=[-10, 30, 6],
            construction_vehicle=[-10, 30, 3],
            bus=[-10, 30, 3],
            trailer=[-10, 30, 3],
            barrier=[-5, 5, 3],
            motorcycle=[-2, 25, 3],
            bicycle=[-2, 15, 2],
            pedestrian=[-1, 10, 2]
        ) if speed_range is None else speed_range
        self.rate = rate
        self.thred_vy_by_vx=thred_vy_by_vx
        self.rate_vy = rate_vy
        self.rate_rotation = rate_rotation
        self.ego_cam = ego_cam

    def interpolating(self, vx, vy, delta_t, box, rot):
        delta_t_max = np.max(delta_t)
        if vy ==0 or vx == 0:
            delta_x = delta_t*vx
            delta_y = np.zeros_like(delta_x)
            rotation_interpolated = np.zeros_like(delta_x)
        else:
            theta = np.arctan2(abs(vy), abs(vx))
            rotation = 2 * theta
            radius = 0.5 * delta_t_max * np.sqrt(vx ** 2 + vy ** 2) / np.sin(theta)
            rotation_interpolated = delta_t / delta_t_max * rotation
            delta_y = radius - radius * np.cos(rotation_interpolated)
            delta_x = radius * np.sin(rotation_interpolated)
            if vy<0:
                delta_y = - delta_y
            if vx<0:
                delta_x = - delta_x
            if np.logical_xor(vx>0, vy>0):
                rotation_interpolated = -rotation_interpolated
        aug = np.zeros((delta_t.shape[0],3,3), dtype=np.float32)
        aug[:, 2, 2] = 1.
        sin = np.sin(-rotation_interpolated)
        cos = np.cos(-rotation_interpolated)
        aug[:,:2,:2] = np.stack([cos,sin,-sin,cos], axis=-1).reshape(delta_t.shape[0], 2, 2)
        aug[:,:2, 2] = np.stack([delta_x, delta_y], axis=-1)

        corner2center = np.eye(3)
        corner2center[0, 2] = -0.5 * box[3]

        instance2ego = np.eye(3)
        yaw = -box[6]
        s = np.sin(yaw)
        c = np.cos(yaw)
        instance2ego[:2,:2] = np.stack([c,s,-s,c]).reshape(2,2)
        instance2ego[:2,2] = box[:2]
        corner2ego = instance2ego @ corner2center
        corner2ego = corner2ego[None, ...]
        if not rot == 0:
            t_rot = np.eye(3)
            s_rot = np.sin(-rot)
            c_rot = np.cos(-rot)
            t_rot[:2,:2] = np.stack([c_rot, s_rot, -s_rot, c_rot]).reshape(2,2)

            instance2ego_ = np.eye(3)
            yaw_ = -box[6] - rot
            s_ = np.sin(yaw_)
            c_ = np.cos(yaw_)
            instance2ego_[:2, :2] = np.stack([c_, s_, -s_, c_]).reshape(2, 2)
            instance2ego_[:2, 2] = box[:2]
            corner2ego_ = instance2ego_ @ corner2center
            corner2ego_ = corner2ego_[None, ...]
            t_rot = instance2ego @ t_rot @ np.linalg.inv(instance2ego)
            aug = corner2ego_ @ aug @ np.linalg.inv(corner2ego_) @ t_rot[None, ...]
        else:
            aug = corner2ego @ aug @ np.linalg.inv(corner2ego)
        return aug

    def __call__(self, results):
        gt_boxes = results['gt_bboxes_3d'].tensor.numpy().copy()
        gt_velocity = gt_boxes[:,7:]
        gt_velocity_norm = np.sum(np.square(gt_velocity), axis=1)
        points = results['points'].tensor.numpy().copy()
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)

        for bid in range(gt_boxes.shape[0]):
            cls = self.cls[results['gt_labels_3d'][bid]]
            points_all = points[point_indices[:, bid]]
            delta_t = np.unique(points_all[:,4])
            aug_rate_cls = self.rate if isinstance(self.rate, float) else self.rate[cls]
            if points_all.shape[0]==0 or \
                    delta_t.shape[0]<3 or \
                    gt_velocity_norm[bid]>0.01 or \
                    cls not in self.speed_range or \
                    np.random.rand() > aug_rate_cls:
                continue

            # sampling speed vx,vy in instance coordinate
            vx = np.random.rand() * (self.speed_range[cls][1] -
                                     self.speed_range[cls][0]) + \
                 self.speed_range[cls][0]
            if np.random.rand() < self.rate_vy:
                max_vy = min(self.speed_range[cls][2]*2, abs(vx) * self.thred_vy_by_vx)
                vy = (np.random.rand()-0.5) * max_vy
            else:
                vy = 0.0
            vx = -vx

            # if points_all.shape[0] == 0 or cls not in self.speed_range or gt_velocity_norm[bid]>0.01 or delta_t.shape[0]<3:
            #     continue
            # vx = 10
            # vy = -2.

            rot = 0.0
            if np.random.rand() < self.rate_rotation:
                rot = (np.random.rand()-0.5) * 1.57

            aug = self.interpolating(vx, vy, delta_t, gt_boxes[bid], rot)

            # update rotation
            gt_boxes[bid, 6] += rot

            # update velocity
            delta_t_max = np.max(delta_t)
            delta_t_max_index = np.argmax(delta_t)
            center = gt_boxes[bid:bid+1, :2]
            center_aug = center @ aug[delta_t_max_index, :2, :2].T + aug[delta_t_max_index, :2, 2]
            vel = (center - center_aug) / delta_t_max
            gt_boxes[bid, 7:] = vel

            # update points
            for fid in range(delta_t.shape[0]):
                points_curr_frame_idxes = points_all[:,4] == delta_t[fid]

                points_all[points_curr_frame_idxes, :2] = \
                    points_all[points_curr_frame_idxes, :2]  @ aug[fid,:2,:2].T + aug[fid,:2, 2:3].T
            points[point_indices[:, bid]] = points_all


        results['points'] = results['points'].new_point(points)
        results['gt_bboxes_3d'] = results['gt_bboxes_3d'].new_box(gt_boxes)
        return results

    def adjust_adj_points(self, adj_points, point_indices_adj, bid, vx, vy, rot, gt_boxes_adj, info_adj, info):
        ts_diff = info['timestamp'] / 1e6 - info_adj['timestamp'] / 1e6
        points = adj_points.tensor.numpy().copy()
        points_all_adj = points[point_indices_adj[:, bid]]
        if points_all_adj.size>0:
            delta_t_adj = np.unique(points_all_adj[:, 4]) + ts_diff
            aug = self.interpolating(vx, vy, delta_t_adj, gt_boxes_adj[bid], rot)
            for fid in range(delta_t_adj.shape[0]):
                points_curr_frame_idxes = points_all_adj[:, 4] == delta_t_adj[fid]- ts_diff
                points_all_adj[points_curr_frame_idxes, :2] = \
                    points_all_adj[points_curr_frame_idxes, :2] @ aug[fid, :2, :2].T + aug[fid, :2, 2:3].T
            points[point_indices_adj[:, bid]] = points_all_adj
        adj_points = adj_points.new_point(points)
        return adj_points

@TRANSFORMS.register_module()
class ImageMask3D(object):

    def __init__(self, failed_number, **kwargs):
        super(ImageMask3D, self).__init__()
        self.number = failed_number

    def __call__(self, input_dict):

        number_list = [0,1,2,3,4,5]

        
        mask_index = random.sample(number_list, self.number)

        for i in mask_index:
            input_dict['img'][i] = 0. * input_dict['img'][i]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str
    
@TRANSFORMS.register_module()
class LidarMask3D(object):
    def __init__(self, angle_range = [-90, 90] , **kwargs):
        super(LidarMask3D, self).__init__()
        self.point_cloud_angle_range = angle_range

    def filter_point_by_angle(self, points):
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        # print(points_numpy[points_numpy[:,1]>0])
        pts_phi = (np.arctan(points_numpy[:, 0] / points_numpy[:, 1]) + (points_numpy[:, 1] < 0) * np.pi + np.pi * 2) % (np.pi * 2) 
        
        pts_phi[pts_phi>np.pi] -= np.pi * 2
        pts_phi = pts_phi/np.pi*180
        
        assert np.all(-180 <= pts_phi) and np.all(pts_phi <= 180)

        filt = np.logical_and(pts_phi>=self.point_cloud_angle_range[0], pts_phi<=self.point_cloud_angle_range[1])
            
        # temp_tensor = points.tensor   
        # temp_tensor[~filt] = temp_tensor[~filt] * 0.0
        # points.tensor = temp_tensor
        # return points
    
        return points[filt]
    def __call__(self, input_dict):
        
        input_dict['points'] = self.filter_point_by_angle(input_dict['points'])
        return input_dict

@TRANSFORMS.register_module()
class RandomFlip3DBEV:
    def __call__(self, data):
        if 'pcd_horizontal_flip' in data:
            flip_horizontal = data['pcd_horizontal_flip']
        else:
            flip_horizontal = random.choice([0, 1])
        if 'pcd_vertical_flip' in data:
            flip_vertical = data['pcd_vertical_flip']
        else:
            flip_vertical = random.choice([0, 1])
        data['flip_horizontal'] = False
        data['flip_vertical'] = False

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if "points" in data:
                data["points"].flip("horizontal")
            if "gt_bboxes_3d" in data:
                data["gt_bboxes_3d"].flip("horizontal")
            if "gt_masks_bev" in data:
                data["gt_masks_bev"] = data["gt_masks_bev"][:, :, ::-1].copy()
            data['flip_horizontal'] = True

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if "points" in data:
                data["points"].flip("vertical")
            if "gt_bboxes_3d" in data:
                data["gt_bboxes_3d"].flip("vertical")
            if "gt_masks_bev" in data:
                data["gt_masks_bev"] = data["gt_masks_bev"][:, ::-1, :].copy()
            data['flip_vertical'] = True

        data["lidar_aug_matrix"][:3, :] = rotation @ data["lidar_aug_matrix"][:3, :]
        return data
    
def transform_matrix_to_seven_vector(T):
    # 提取平移部分
    translation = T[:3, 3]
    
    # 提取旋转部分
    R = T[:3, :3]
    
    # 计算四元数
    q_w = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    q_x = (R[2, 1] - R[1, 2]) / (4 * q_w)
    q_y = (R[0, 2] - R[2, 0]) / (4 * q_w)
    q_z = (R[1, 0] - R[0, 1]) / (4 * q_w)
    
    # 创建 7x1 矩阵
    result = np.array([translation[0], translation[1], translation[2], q_w, q_x, q_y, q_z]).reshape(7, 1)
    
    return result
def remove_close(points, radius=1.0):
    """Removes point too close within a certain radius from origin.

    Args:
        points (np.ndarray): Sweep points.
        radius (float): Radius below which points are removed.
            Defaults to 1.0.

    Returns:
        np.ndarray: Points after removing.
    """
    if isinstance(points, np.ndarray):
        points_numpy = points
    elif isinstance(points, BasePoints):
        points_numpy = points.tensor.numpy()
    else:
        raise NotImplementedError
    x_filt = np.abs(points_numpy[:, 0]) < radius
    y_filt = np.abs(points_numpy[:, 1]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    return points[not_close]
def load_points(pts_filename):
    """Private function to load point clouds data.

    Args:
        pts_filename (str): Filename of point clouds data.

    Returns:
        np.ndarray: An array containing point clouds data.
    """
    points = np.fromfile(pts_filename, dtype=np.float32)
    return points
@TRANSFORMS.register_module()
class corruption_LC(object):
    def __init__(self, cor_type, severity, seed=2024):
        super(corruption_LC, self).__init__()
        self.type = cor_type
        self.severity = severity
        if self.type == 'snow': # takes around 10 hours in 8*A100 for eval suggest to use https://github.com/SysCV/LiDAR_snow_sim 64.03
            self.snow_sim = ImageAddSnow(severity, seed)
        elif self.type == 'rain': # takes around 10 hours in 8*A100 for eval (other corruptions is fast) 69.09
            self.rain_sim = ImageAddRain(severity, seed)
        elif self.type == 'fog': # 71.13 69.49 65.52 62.02 40.82 avg: 61.80
            # refer to https://github.com/MartinHahner/LiDAR_fog_sim
            self.fog_sim = ImageAddFog(severity, seed)
        elif self.type == 'sunlight': # 71.90 71.27 70.90 70.37 69.95 avg: 70.88
            self.sun_sim = ImagePointAddSun(severity)
        elif self.type == 'density': # 72.43 72.32 72.26 72.10 71.94 avg: 72.21
            pass
        elif self.type == 'cutout': # 71.98 71.71 71.25 70.35 69.29 avg: 70.92
            pass
        elif self.type == 'lidar_crosstalk': # avg: 71.18
            pass
        elif self.type == 'fov_lost': # 38.80 34.06 29.52 24.64 18.75 avg: 29.15
            pass
        elif self.type == 'gaussian_l': # 72.38 71.49 69.88 67.22 63.21 avg: 68.84
            pass
        elif self.type == 'uniform_l': # 72.51 72.23 71.80 72.22 70.32 avg: 71.82
            pass
        elif self.type == 'impulse_l': # 72.21 72.13 72.11 72.00 71.69 avg: 72.03
            pass
        elif self.type == 'gaussian_c': # 69.82 67.75 65.61 63.95 62.57 avg: 65.94
            self.gaussian_sim = ImageAddGaussianNoise(severity, seed)
        elif self.type == 'uniform_c': # 71.49 70.37 68.58 66.55 64.52 avg: 68.30
            self.uniform_sim = ImageAddUniformNoise(severity)
        elif self.type == 'impulse_c': # 68.78 66.52 65.30 63.71 62.45 avg: 65.35
            self.impulse_sim = ImageAddImpulseNoise(severity, seed)
        elif self.type == 'compensation': # 44.83 42.63 39.52 36.12 32.43 avg: 39.12 # set num_workers to 0, and persistent_workers to False
            pass
        elif self.type == 'motion_object': # 67.47 59.04 47.59 41.10 35.20 avg: 50.08
            self.object_motion_sim_frontback = ImageBBoxMotionBlurFrontBack(
                severity=severity,
                corrput_list=[0.02 * i for i in range(1, 6)],
            )
            self.object_motion_sim_leftright = ImageBBoxMotionBlurLeftRight(
                severity=severity,
                corrput_list=[0.02 * i for i in range(1, 6)],
            )
        elif self.type == 'motion_blur': # 70.97 68.81 66.76 65.02 63.59 avg: 67.03
            self.motion_blur_sim_frontback = ImageMotionBlurFrontBack(
                severity=severity,
                corrput_list=[0.02 * i for i in range(1, 6)],
            )
            self.motion_blur_sim_leftright = ImageMotionBlurLeftRight(
                severity=severity,
                corrput_list=[0.02 * i for i in range(1, 6)],
            )
        elif self.type == 'local_density': # 72.36 72.13 71.74 71.30 70.64 avg: 71.63
            pass 
        elif self.type == 'local_cutout': # 70.97 69.89 68.46 66.32 63.33 avg: 67.79
            pass 
        elif self.type == 'local_gaussian': # 72.74 72.27 71.07 69.52 67.30 avg: 70.58
            pass 
        elif self.type == 'local_uniform': # 72.70 72.76 72.60 72.03 71.34 avg: 72.29
            pass
        elif self.type == 'local_impulse': # 72.42 72.39 72.33 72.26 72.14 avg: 72.31
            pass
        elif self.type == 'shear': # 66.12 63.17 59.36 54.84 50.43 avg: 58.78
            self.shear_sim = ImageBBoxOperation(self.severity)
        elif self.type == 'scale': # 71.35 69.20 66.71 63.45 59.63 avg: 66.07
            self.scale_sim = ImageBBoxOperation(self.severity)
        elif self.type == 'rotation': # 71.87 70.83 69.54 68.22 66.98 avg: 69.49
            self.rotation_sim = ImageBBoxOperation(self.severity)
        elif self.type == 'spatial_alignment': # 72.40 71.79 71.15 70.60 69.98 avg: 71.18
            pass
        elif self.type == 'temporal_alignment': # 66.57 64.14 58.66 47.31 39.64 avg: 55.26  68.09 65.63 60.05 48.58 41.12 avg: 56.64
            pass                                # we argue the original code in https://github.com/thu-ml/3D_Corruptions_AD/ has problem
                                                # so we make a new implement according to the original paper.
                                                

    def __call__(self, input_dict):
        imgs = input_dict['img']
        pts = input_dict['points'].tensor
        lidar2img = input_dict['lidar2img']
        if self.type == 'snow':
            
            img_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]] # bgr-> rgb
                img_i = self.snow_sim(img_i)
                img_aug_list.append(img_i[:, :, [2, 1, 0]]) # rgb -> bgr

            pts_aug = snow_sim(pts[:, :4], self.severity)
            pts_aug = torch.cat([torch.from_numpy(pts_aug[:, :4]), pts[:, 4:5]], dim=-1).float()

            input_dict['img'] = img_aug_list
            input_dict['points'].tensor = pts_aug

        elif self.type == 'rain':
            
            img_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]] # bgr-> rgb
                img_i = self.rain_sim(img_i)
                img_aug_list.append(img_i[:, :, [2, 1, 0]]) # rgb -> bgr

            pts_aug = rain_sim(pts[:, :4], self.severity)
            pts_aug = torch.cat([torch.from_numpy(pts_aug[:, :4]), pts[:, 4:5]], dim=-1).float()

            input_dict['img'] = img_aug_list
            input_dict['points'].tensor = pts_aug

        elif self.type == 'fog':
            
            img_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]] # bgr-> rgb
                img_i = self.fog_sim(img_i)
                img_aug_list.append(img_i[:, :, [2, 1, 0]]) # rgb -> bgr

            pts_aug = torch.from_numpy(fog_sim(pts.numpy(), self.severity)).float()

            input_dict['img'] = img_aug_list
            input_dict['points'].tensor = pts_aug

        elif self.type == 'sunlight': 
            img_0 = imgs[0][:, :, [2, 1, 0]] # bgr-> rgb
            lidar2img_0 = lidar2img[0]
            img_0, pts_aug = self.sun_sim(
                image=img_0,
                points=pts,
                lidar2img=lidar2img_0,
            )
            img_0 = img_0[:, :, [2, 1, 0]] # rgb -> bgr
            imgs[0] = img_0
            input_dict['img'] = imgs
            input_dict['points'].tensor = pts_aug

        elif self.type == 'density':
            pts_aug = density_dec_global(pts, self.severity)
            input_dict['points'].tensor = pts_aug

        elif self.type == 'cutout':
            pts_aug = cutout_local(pts.numpy(), self.severity)
            input_dict['points'].tensor = torch.from_numpy(pts_aug)

        elif self.type == 'lidar_crosstalk':
            pts_aug = lidar_crosstalk_noise(pts.numpy(), self.severity)
            input_dict['points'].tensor = torch.from_numpy(pts_aug)

        elif self.type == 'fov_lost':
            pts_aug = fov_filter(pts.numpy(), self.severity)
            input_dict['points'].tensor = torch.from_numpy(pts_aug)

        elif self.type == 'gaussian_l':
            pts_aug = gaussian_noise(pts.numpy(), self.severity)
            input_dict['points'].tensor = torch.from_numpy(pts_aug)

        elif self.type == 'uniform_l':
            pts_aug = uniform_noise(pts.numpy(), self.severity)
            input_dict['points'].tensor = torch.from_numpy(pts_aug)

        elif self.type == 'impulse_l':
            pts_aug = impulse_noise(pts.numpy(), self.severity)
            input_dict['points'].tensor = torch.from_numpy(pts_aug)

        elif self.type == 'gaussian_c':
            img_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]] # bgr-> rgb
                img_i = self.gaussian_sim(img_i)
                img_aug_list.append(img_i[:, :, [2, 1, 0]]) # rgb -> bgr

            input_dict['img'] = img_aug_list

        elif self.type == 'uniform_c':
            img_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]] # bgr-> rgb
                img_i = self.uniform_sim(img_i)
                img_aug_list.append(img_i[:, :, [2, 1, 0]]) # rgb -> bgr

            input_dict['img'] = img_aug_list

        elif self.type == 'impulse_c':
            img_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]] # bgr-> rgb
                img_i = self.impulse_sim(img_i)
                img_aug_list.append(img_i[:, :, [2, 1, 0]]) # rgb -> bgr

            input_dict['img'] = img_aug_list

        elif self.type == 'compensation': # set num_workers to 0, and persistent_workers to False
            if 'lidar_sweeps' in input_dict.keys():
                ego_pose_1 = transform_matrix_to_seven_vector(np.array(input_dict['ego2global'])).squeeze()
                ego_pose_2 = transform_matrix_to_seven_vector(np.array(input_dict['lidar_points']['lidar2ego'])).squeeze()

                sec_sweep_1 = transform_matrix_to_seven_vector(np.array(input_dict['lidar_sweeps'][0]['ego2global'])).squeeze()
                sec_sweep_2 = transform_matrix_to_seven_vector(np.dot(np.array(input_dict['lidar_sweeps'][0]['lidar_points']['lidar2ego']), \
                                                    np.linalg.inv(np.array(input_dict['lidar_sweeps'][0]['lidar_points']['lidar2sensor'])))).squeeze()
                pc_pose = np.stack([ego_pose_1, ego_pose_2, sec_sweep_1, sec_sweep_2])
                pts_aug = fulltrajectory_noise(pts[:, :3].numpy(), pc_pose, self.severity)
                pts[:, :3] = pts_aug
                input_dict['points'].tensor = pts
            else:
                pass
        elif self.type == 'motion_object':
            bboxes_corners = input_dict['eval_ann_info']['gt_bboxes_3d'].corners
            bboxes_centers = input_dict['eval_ann_info']['gt_bboxes_3d'].center
            img_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]] # bgr-> rgb
                if i % 3 == 0:
                    img_i = self.object_motion_sim_frontback(
                        image=img_i,
                        bboxes_centers=bboxes_centers,
                        bboxes_corners=bboxes_corners,
                        lidar2img=lidar2img[i],)
                else:
                    img_i = self.object_motion_sim_leftright(
                        image=img_i,
                        bboxes_centers=bboxes_centers,
                        bboxes_corners=bboxes_corners,
                        lidar2img=lidar2img[i],)
                img_aug_list.append(img_i[:, :, [2, 1, 0]]) # rgb -> bgr
            input_dict['img'] = img_aug_list
            pts_aug = moving_noise_bbox(pts.numpy(), self.severity, [input_dict['eval_ann_info']['gt_bboxes_3d']])
            input_dict['points'].tensor = torch.from_numpy(pts_aug)
        elif self.type == 'motion_blur':
            img_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]] # bgr-> rgb
                if i % 3 == 0:
                    img_i = self.motion_blur_sim_frontback(
                        image=img_i)
                else:
                    img_i = self.motion_blur_sim_leftright(
                        image=img_i)
                img_aug_list.append(img_i[:, :, [2, 1, 0]]) # rgb -> bgr
            input_dict['img'] = img_aug_list

        elif self.type == 'local_density':
            pts_aug = density_dec_bbox(pts.numpy(), self.severity, [input_dict['eval_ann_info']['gt_bboxes_3d']])
            input_dict['points'].tensor = torch.from_numpy(pts_aug)
        elif self.type == 'local_cutout':
            pts_aug = cutout_bbox(pts.numpy(), self.severity, [input_dict['eval_ann_info']['gt_bboxes_3d']])
            input_dict['points'].tensor = torch.from_numpy(pts_aug)
        elif self.type == 'local_gaussian':
            pts_aug = gaussian_noise_bbox(pts.numpy(), self.severity, [input_dict['eval_ann_info']['gt_bboxes_3d']])
            input_dict['points'].tensor = torch.from_numpy(pts_aug)
        elif self.type == 'local_uniform':
            pts_aug = uniform_noise_bbox(pts.numpy(), self.severity, [input_dict['eval_ann_info']['gt_bboxes_3d']])
            input_dict['points'].tensor = torch.from_numpy(pts_aug)
        elif self.type == 'local_impulse':
            pts_aug = impulse_noise_bbox(pts.numpy(), self.severity, [input_dict['eval_ann_info']['gt_bboxes_3d']])
            input_dict['points'].tensor = torch.from_numpy(pts_aug)

        elif self.type == 'shear':
            pts_aug = shear_bbox(pts.numpy(), self.severity, [input_dict['eval_ann_info']['gt_bboxes_3d']])
            input_dict['points'].tensor = torch.from_numpy(pts_aug).float()

            bboxes_corners = input_dict['eval_ann_info']['gt_bboxes_3d'].corners
            bboxes_centers = input_dict['eval_ann_info']['gt_bboxes_3d'].center
            img_aug_list = []
            c = [0.05, 0.1, 0.15, 0.2, 0.25][self.severity - 1]
            b = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
            d = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
            e = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
            f = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
            transform_matrix = torch.tensor([
                [1, 0, b],
                [d, 1, e],
                [f, 0, 1]
            ]).float()
            
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]] # bgr-> rgb
                img_i = self.shear_sim(
                    image=img_i,
                    transform_matrix=transform_matrix,
                    bboxes_centers=bboxes_centers,
                    bboxes_corners=bboxes_corners,
                    lidar2img=lidar2img[i],)
                img_aug_list.append(img_i[:, :, [2, 1, 0]]) # rgb -> bgr
            input_dict['img'] = img_aug_list

        elif self.type == 'scale':
            pts_aug = scale_bbox(pts.numpy(), self.severity, [input_dict['eval_ann_info']['gt_bboxes_3d']])
            input_dict['points'].tensor = torch.from_numpy(pts_aug).float()

            bboxes_corners = input_dict['eval_ann_info']['gt_bboxes_3d'].corners
            bboxes_centers = input_dict['eval_ann_info']['gt_bboxes_3d'].center
            img_aug_list = []
            
            c = [0.1, 0.2, 0.3, 0.4, 0.5][self.severity - 1]
            a = b = d = 1
            r = np.random.randint(0, 3)
            t = np.random.choice([-1, 1])
            a += c * t
            b += c * t
            d += c * t

            transform_matrix = torch.tensor([
                [a, 0, 0],
                [0, b, 0],
                [0, 0, d],
            ]).float()
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]] # bgr-> rgb
                img_i = self.scale_sim(
                    image=img_i,
                    transform_matrix=transform_matrix,
                    bboxes_centers=bboxes_centers,
                    bboxes_corners=bboxes_corners,
                    lidar2img=lidar2img[i],)
                img_aug_list.append(img_i[:, :, [2, 1, 0]]) # rgb -> bgr
            input_dict['img'] = img_aug_list

        elif self.type == 'rotation':
            pts_aug = rotation_bbox(pts.numpy(), self.severity, [input_dict['eval_ann_info']['gt_bboxes_3d']])
            input_dict['points'].tensor = torch.from_numpy(pts_aug).float()

            bboxes_corners = input_dict['eval_ann_info']['gt_bboxes_3d'].corners
            bboxes_centers = input_dict['eval_ann_info']['gt_bboxes_3d'].center
            img_aug_list = []
            theta_base = [4, 8, 12, 16, 20][self.severity - 1]
            theta_degree = np.random.uniform(theta_base - 2, theta_base + 2) * np.random.choice([-1, 1])

            theta = theta_degree / 180 * np.pi
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            transform_matrix = torch.tensor([
                        [cos_theta, sin_theta, 0],
                        [-sin_theta, cos_theta, 0],
                        [0, 0, 1],
                    ]).float()
            
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]] # bgr-> rgb
                img_i = self.rotation_sim(
                    image=img_i,
                    transform_matrix=transform_matrix,
                    bboxes_centers=bboxes_centers,
                    bboxes_corners=bboxes_corners,
                    lidar2img=lidar2img[i],)
                img_aug_list.append(img_i[:, :, [2, 1, 0]]) # rgb -> bgr
            input_dict['img'] = img_aug_list
        elif self.type == 'spatial_alignment':
            noise_pose = []
            for i in range(6):
                noise_pose.append(spatial_alignment_noise(lidar2img[i], self.severity))
            input_dict['lidar2img'] = noise_pose
        elif self.type == 'temporal_alignment':
            frame = temporal_alignment_noise(self.severity)
            random_number = np.random.random()
            if random_number<0.5: # image stuck
                if len(input_dict['cam_sweeps'][0]) == 0:
                    pass
                else:    
                    image_aug_bgr = []
                    for i in range(6):
                        filename = 'data/nuscenes/'+input_dict['cam_sweeps'][i][frame-1]['filename']
                        img_temp = mmcv.imread(filename)
                        image_aug_bgr.append(img_temp)
                    input_dict['img'] = image_aug_bgr

            else: # lidar stuck
                if 'lidar_sweeps' not in input_dict.keys():
                    pass
                else:
                    lidar_info = input_dict['lidar_sweeps']
                    if len(lidar_info) < frame+10:
                        pass
                    else:
                        points_sweep_list = []
                        for idx_f in range(frame-1, frame+10):
                            sweep = lidar_info[idx_f]
                            points_sweep = load_points(
                                sweep['lidar_points']['lidar_path'])
                            points_sweep = np.copy(points_sweep).reshape(-1, 5)

                            points_sweep = remove_close(points_sweep)
                            sweep_ts = sweep['timestamp']
                            ts = input_dict['timestamp']
                            lidar2sensor = np.array(sweep['lidar_points']['lidar2sensor'])
                            points_sweep[:, :
                                        3] = points_sweep[:, :3] @ lidar2sensor[:3, :3]
                            points_sweep[:, :3] -= lidar2sensor[:3, 3]
                            points_sweep[:, 4] = ts - sweep_ts
                            points_sweep_list.append(points_sweep)
                        
                        points_sweep_list = np.concatenate(points_sweep_list, axis=0)
                        input_dict['points'].tensor = torch.from_numpy(points_sweep_list)
        return input_dict