from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F
import time
import json
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from mmdet3d.models.test_time_augs import merge_aug_bboxes_3d
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from .ops import Voxelization

def create_2D_grid(self, x_size, y_size):
    meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
    # NOTE: modified
    batch_x, batch_y = torch.meshgrid(
        *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
    batch_x = batch_x + 0.5
    batch_y = batch_y + 0.5
    coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
    coord_base = coord_base
    return coord_base

@MODELS.register_module()
class CoreNet(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        DisMoudule: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None
        self.DisMoudule = MODELS.build(
            DisMoudule) if DisMoudule is not None else None
        
        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)
        self.bbox_head = MODELS.build(bbox_head)

        self.init_weights()

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        mode,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W).contiguous()

        # with torch.autocast(device_type='cuda', dtype=torch.float32):
        #     x = self.view_transform(
        #         x,
        #         points,
        #         lidar2image,
        #         camera_intrinsics,
        #         camera2lidar,
        #         img_aug_matrix,
        #         lidar_aug_matrix,
        #         img_metas,
        #         mode,
        #     )
        # return x, None

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x, gt_depth, pred_depth = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
                mode,
            )
        if mode == 'train':
            with torch.autocast('cuda', enabled=False):
                depth_loss = self.view_transform.get_depth_loss(gt_depth.squeeze(2), pred_depth)
        
        else:
            depth_loss = None
        return x, depth_loss

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        # torch.cuda.synchronize() #在测fps时去掉voxel的生成时间
        # start_time = time.perf_counter()
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points) # [N, 5] [N, 4]
            batch_size = coords[-1, 0] + 1
        # with open("voxel_time.json","w") as f:
        #     torch.cuda.synchronize()
        #     json.dump(time.perf_counter() - start_time,f)
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, _ = self.extract_feat(batch_inputs_dict, batch_input_metas, mode='inference')

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)
        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res        
      
    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        mode,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
    
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature, depth_loss = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas,mode)

        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        if imgs is not None:
            img_spec, pts_spec, share_spec = self.DisMoudule(img_feature, pts_feature)

        if self.fusion_layer is not None:
            fuse_feat = self.fusion_layer(share_spec)
        else:
            fuse_feat = pts_feature
        
        # SECOND Head
        x = self.pts_backbone(fuse_feat)
        x = self.pts_neck(x)

        if imgs is not None: # fusion stage
            return [x[0], img_spec, pts_spec], depth_loss
        else: # pre-train stage
            return x, depth_loss

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, depth_loss = self.extract_feat(batch_inputs_dict, batch_input_metas, mode='train')

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
        losses.update(bbox_loss)
        losses['depth_loss'] = depth_loss
        return losses

        
