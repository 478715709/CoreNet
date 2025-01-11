<div align="center">
<!-- <h1>RCTrans</h1> -->
<h3>[INFFUS 2025] CoreNet: Conflict Resolution Network for point-pixel misalignment and sub-task suppression of 3D LiDAR-camera object detection</h3>
<h4>Yiheng Li, Yang Yang and Zhen Lei<h4>
<h5>MAIS&CASIA, UCAS<h5>
</div>

## Introduction

This repository is an official implementation of CoreNet.

## News
- [2025/1/11] Codes are released.
- [2025/1/7] Camera Ready version is released [paper](https://www.sciencedirect.com/science/article/pii/S1566253524006742?via%3Dihub).
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
python tools/create_data_nusc.py --root-path ./data/nuscenes --out-dir ./data --extra-tag nuscenes_radar --version v1.0
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
|   ├── nuscenes_infos_test.pkl
|   ├── nuscenes_infos_val.pkl
|   ├── nuscenes_infos_train.pkl
```
## Train
You should train lidar brach first and merge the weight with image backbones. We give our pre-trained merged weight [here](https://drive.usercontent.google.com/download?id=1DzEw7MwVuBYLDD-e-8LEYwS_ulnroxJD&export=download&authuser=0&confirm=t&uuid=5d163f50-7a28-473d-9f70-817c97963f8c&at=AIrpjvMMlEhoWjOzWh9KOAZ-OgPm:1736602091374).
The pre-train steps of our methods is similar to BEVFusion, the only difference is that we use velocity augmentation from DAL.
You can then conduct multi-modal training via the following command.

```
bash tools/dist_train.sh projects/corenet/configs/corenet_lidar-cam-res50.py 8 --work-dir ${LOG_DIR}
```
The pre-trianed best weight of CoreNet is [here](https://drive.usercontent.google.com/download?id=1F5pKBZEXT40y0qru-TLFPga8z3HhhJgZ&export=download&authuser=0&confirm=t&uuid=5f42ed2d-4f01-4d88-b761-49b6c7bd5f4a&at=AIrpjvMOJJY4JrdbyhrRczMM8j_j:1736602154582).

## Test
```
bash tools/dist_test.sh projects/corenet/configs/corenet_lidar-cam.py pretrain_model/val-best.pth 8
```

## Acknowledgements
We thank these great works and open-source codebases: BEVFusion, IS-Fusion, SparseFusion, DAL, MMDetection3D.

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