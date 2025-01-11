_base_ = [
    './corenet_lidar.py'
]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None
randomness = dict(seed = 2023)
max_epoch = 6
img_scale = [384, 1056]

model = dict(
    type='CoreNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),

    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    view_transform=dict(
        type='DualTransform',
        in_channels=256,
        out_channels=80,
        final_channels=256,
        image_size=img_scale,
        feature_size=[int(img_scale[0]/8), int(img_scale[1]/8)],
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2,
        use_drf=False),
    DisMoudule = dict(
        type='DisentangleMoudule', pts_dim=256, img_dim=256),
    fusion_layer=dict(
        type='LRF', in_channels=[128, 128], out_channels=256),
    bbox_head=dict(
        type='CoreNetHead',
        num_proposals=300,  # 300 proposals
        img_fuse = dict(type='dimconv_ts', indim=[128,128] , outdim=128),
        pts_fuse = dict(type='dimconv_ts', indim=[128,128] , outdim=128),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.5),
        aux_pre = True,
        aux_loss = True))

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ImageAug3D',
        final_dim=img_scale,
        resize_lim=[0.56, 0.825],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.7,
        fixed_prob=True),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    # dict(
    #     type='corruption_LC',
    #     cor_type= 'rain',
    #     severity = 3, # [1,2,3,4,5]
    # ),
    dict(
        type='ImageAug3D',
        final_dim=img_scale,
        resize_lim=[0.72, 0.72],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0, 0],
        rand_flip=False,
        is_train=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale= (img_scale[0], img_scale[1]),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='BEVFusionGlobalRotScaleTrans',
                scale_ratio_range=[1.0, 1.0],
                rot_range=[0, 0],
                translation_std=[0,0,0]),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
        ]),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        dataset=dict(pipeline=train_pipeline, modality=input_modality)))
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, modality=input_modality))
test_dataloader = val_dataloader

# runtime settings

train_cfg = dict(by_epoch=True, max_epochs=max_epoch, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# schedule 6
lr = 0.0001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=max_epoch,
        end=max_epoch,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95,
        begin=0,
        end=0.4*max_epoch,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        begin=0.4*max_epoch,
        end=max_epoch,
        by_epoch=True,
        convert_to_iter_based=True)
]
    
auto_scale_lr = dict(enable=False, base_batch_size=16)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))
del _base_.custom_hooks
del _base_.db_sampler
# find_unused_parameters = True
# load pre-trained model
load_from='pretrain_model/voxel0075_vel_nuimage_train.pth' # pre_trained model

'''
# 4.5fps in NVIDIA A100 params: 87.5M mem_cost: 9207
mAP: 0.7255
mATE: 0.2669
mASE: 0.2535
mAOE: 0.2890
mAVE: 0.1896
mAAE: 0.1811
NDS: 0.7447

Per-class results:
                AMOTA   AMOTP   RECALL  MOTAR   GT      MOTA    MOTP    MT      ML      FAF     TP      FP      FN      IDS     FRAG    TID     LGD
bicycle         0.704   0.472   0.711   0.815   1993    0.578   0.229   93      53      17.6    1413    262     576     4       6       0.26    0.43
bus             0.896   0.423   0.895   0.887   2112    0.789   0.402   89      10      13.0    1879    213     222     11      11      0.17    0.48
car             0.860   0.277   0.870   0.842   58317   0.730   0.232   2861    656     138.7   50579   7997    7561    177     189     0.15    0.31
motorcy         0.773   0.448   0.840   0.810   1977    0.669   0.291   96      20      22.3    1633    310     317     27      22      0.38    0.62
pedestr         0.751   0.374   0.836   0.768   25423   0.629   0.270   1213    336     110.2   20835   4843    4175    413     91      0.25    0.62
trailer         0.598   0.772   0.642   0.787   2425    0.503   0.542   72      51      32.0    1551    331     869     5       12      0.25    0.77
truck           0.757   0.474   0.846   0.748   9650    0.632   0.343   404     101     53.4    8147    2053    1484    19      37      0.23    0.47

Aggregated results:
AMOTA   0.763
AMOTP   0.463
RECALL  0.806
MOTAR   0.808
GT      14556
MOTA    0.647
MOTP    0.330
MT      4828
ML      1227
FAF     55.3
TP      86037
FP      16009
FN      15204
IDS     656
FRAG    368
TID     0.24
LGD     0.53

'''