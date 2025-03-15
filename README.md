<div align="center">
<!-- <h1>RCTrans</h1> -->
<h3>[INFFUS 2025] CoreNet: Conflict Resolution Network for point-pixel misalignment and sub-task suppression of 3D LiDAR-camera object detection</h3>
<h4>Yiheng Li, Yang Yang and Zhen Lei<h4>
<h5>MAIS&CASIA, UCAS<h5>
</div>

## Introduction

This repository is an official implementation of CoreNet.

## News
- [2025/2/23] Implement details of robustness analysis are released.
- [2025/1/11] Codes are released.
- [2025/1/7] Camera Ready version is released [arxiv](https://arxiv.org/abs/2501.06550).
- [2024/12/18] CoreNet is accepted by Informantion Fusion 2025 🎉🎉.


## Environment Setting
```
conda create -n CoreNet python=3.8
conda activate CoreNet
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install mmengine==0.10.4
pip install mmcv==2.0.1
pip install -r requirements.txt
pip install "opencv-python-headless<4.3"
python3 -m pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ mmdet==3.1.0
pip install pyquaternion==0.9.9
pip install trimesh==3.23.0
pip install lyft-dataset-sdk
pip install nuscenes-devkit==1.1.10
pip install einops
pip install timm
pip install spconv-cu111
pip install h5py
pip install imagecorruptions
pip install distortion
pip install PyMieScatt 
pip install -v -e .
python projects/corenet/setup.py develop
```

## Data Preparation
```
python tools/create_data.py --root-path ./data/nuscenes --out-dir ./data --extra-tag nuscenes --version v1.0
```
Floder structure
```
CoreNet
├── projects/
├── mmdetection3d/
├── tools/
├── ckpts/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
│   ├── nuscenes_database
│   ├── nuscenes_infos_train.pkl
|   ├── nuscenes_infos_test.pkl
|   ├── nuscenes_infos_val.pkl
|   ├── nuscenes_dbinfos_train.pkl
```
## Train
You should train lidar brach first and merge the weight with image backbones. We give our pre-trained merged weight [here](https://drive.usercontent.google.com/download?id=1DzEw7MwVuBYLDD-e-8LEYwS_ulnroxJD&export=download&authuser=0&confirm=t&uuid=5d163f50-7a28-473d-9f70-817c97963f8c&at=AIrpjvMMlEhoWjOzWh9KOAZ-OgPm:1736602091374).
The pre-train steps of our methods is similar to BEVFusion, the only difference is that we use velocity augmentation from DAL.
You can then conduct multi-modal training via the following command.

```
bash tools/dist_train.sh projects/corenet/configs/corenet_lidar-cam.py 8 --work-dir ${LOG_DIR}
```
The pre-trianed best weight of CoreNet is [here](https://drive.usercontent.google.com/download?id=1F5pKBZEXT40y0qru-TLFPga8z3HhhJgZ&export=download&authuser=0&confirm=t&uuid=5f42ed2d-4f01-4d88-b761-49b6c7bd5f4a&at=AIrpjvMOJJY4JrdbyhrRczMM8j_j:1736602154582).

The log of training process is [here](https://drive.usercontent.google.com/download?id=1nqzckVEL2DWO9EevGnIk-HczfMHmL2em&export=download&authuser=0&confirm=t&uuid=a296e4f6-50fa-4e27-9a77-4fcce976f23a&at=AIrpjvMlK1Q-mIpBH1G4b5ettrXX:1738640762151).

## Test
```
bash tools/dist_test.sh projects/corenet/configs/corenet_lidar-cam.py pretrain_model/val-best.pth 8
```

## Results of nuscenes-C (Table-10)
```
Release the corruption_LC in test_pipeline in corenet_lidar-cam.py for evaluate. The original codes is in corenet/transforms_3d.py
```

## Results of robustness analysis (Table-8 and Table-9)
```
Modify the filter_eval_boxes function in nuscenes-devkit to filter objects as following:
# for table-8
description = nusc.get('scene",nusc.get('sample', sample token)['scene token'])['description']
# if('Rain' in description) or ('rain' in description):
#   eval_boxes.boxes[sample token]=[] # sunny

# if('Rain' not in description) and ('rain' not in description):
#   eval_boxes.boxes[sample token] = [] # rainy

# if('Night'in description) or ('night' in description):
#   eval_boxes.boxes[sample token]=[] # day

# if('Night' not in description) and ('night' not in description):
#   eval_boxes.boxes[sample token] = [] # night

# for table-9 
# eval_boxes.boxes[sample_token]= [box for box in eval_boxes[sample_token] if box.ego_dist< 20] # near

# eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if (20 < box.ego_dist and box.ego_dist < 30)] # middle

# eval_boxes.boxes[sample_token]= [box for box in eval_boxes[sample_token] if box.ego_dist > 30] # far

# eval_boxes.boxes[sample_token]= [box for box in eval_boxes[sample_token] if max(box.size)<4] # small

# eval_boxes.boxes[sample_token]= [box for box in eval_boxes[sample_token] if max(box.size)>4] # large
```

## Acknowledgements
We thank these great works and open-source codebases: [BEVFusion](https://github.com/mit-han-lab/bevfusion), [IS-Fusion](https://github.com/yinjunbo/IS-Fusion), [SparseFusion](https://github.com/yichen928/SparseFusion), [DAL](https://github.com/HuangJunJie2017/BEVDet), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

## Citation
```
@article{li2025corenet,
  title={CoreNet: Conflict Resolution Network for point-pixel misalignment and sub-task suppression of 3D LiDAR-camera object detection},
  author={Li, Yiheng and Yang, Yang and Lei, Zhen},
  journal={Information Fusion},
  pages={102896},
  year={2025},
  publisher={Elsevier}
}
```