"""
BraTS2023数据集预处理脚本
"""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

class BraTS2023Dataset(Dataset):
    """BraTS2023数据集类"""
    
    def __init__(self, data_dir, split='train', modalities=['FLAIR', 'T1', 'T1c', 'T2'], 
                 image_size=(128, 128, 128), transform=None):
        self.data_dir = data_dir
        self.split = split
        self.modalities = modalities
        self.image_size = image_size
        self.transform = transform
        
        # 获取样本列表
        self.samples = self._get_sample_list()
        
    def _get_sample_list(self):
        """获取样本列表"""
        # 这里需要根据实际的数据集结构实现
        # 假设每个样本是一个文件夹，包含模态文件和分割文件
        samples = []
        
        for patient_id in os.listdir(self.data_dir):
            sample_path = os.path.join(self.data_dir, patient_id)
            if os.path.isdir(sample_path):
                samples.append(patient_id)
        
        # 划分训练/验证/测试集
        np.random.seed(42)
        np.random.shuffle(samples)
        
        if self.split == 'train':
            samples = samples[:int(0.7 * len(samples))]
        elif self.split == 'val':
            samples = samples[int(0.7 * len(samples)):int(0.8 * len(samples))]
        else:  # test
            samples = samples[int(0.8 * len(samples)):]
        
        return samples
    
    def load_nifti(self, filepath):
        """加载nifti文件"""
        img = nib.load(filepath)
        data = img.get_fdata()
        
        # 标准化到[0, 1]
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        # 调整大小
        data = self._resize_volume(data, self.image_size)
        
        return data
    
    def _resize_volume(self, volume, target_size):
        """调整3D体积大小"""
        # 这里可以使用插值方法，简化版使用裁剪/填充
        # 实际应用中应该使用3D插值
        current_shape = volume.shape
        target_shape = target_size
        
        # 计算填充/裁剪
        pad_width = []
        crop_start = []
        
        for i in range(3):
            if current_shape[i] < target_shape[i]:
                # 需要填充
                pad_before = (target_shape[i] - current_shape[i]) // 2
                pad_after = target_shape[i] - current_shape[i] - pad_before
                pad_width.append((pad_before, pad_after))
                crop_start.append(0)
            else:
                # 需要裁剪
                crop_before = (current_shape[i] - target_shape[i]) // 2
                pad_width.append((0, 0))
                crop_start.append(crop_before)
        
        # 应用填充
        if any(p[0] > 0 or p[1] > 0 for p in pad_width):
            volume = np.pad(volume, pad_width, mode='constant')
        
        # 应用裁剪
        crop_slices = tuple(slice(crop_start[i], crop_start[i] + target_shape[i]) 
                           for i in range(3))
        volume = volume[crop_slices]
        
        return volume
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        patient_id = self.samples[idx]
        patient_path = os.path.join(self.data_dir, patient_id)
        
        # 加载模态
        modalities_data = []
        for modality in self.modalities:
            # 构建文件路径
            file_pattern = f"{patient_id}_{modality}.nii.gz"
            filepath = os.path.join(patient_path, file_pattern)
            
            if os.path.exists(filepath):
                modality_data = self.load_nifti(filepath)
            else:
                # 如果文件不存在，用零填充
                modality_data = np.zeros(self.image_size)
            
            modalities_data.append(modality_data)
        
        # 加载分割标签
        seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
        if os.path.exists(seg_path):
            segmentation = self.load_nifti(seg_path)
            # 将分割标签转换为类别
            segmentation = self._convert_segmentation_labels(segmentation)
        else:
            segmentation = np.zeros(self.image_size)
        
        # 转换为张量
        modalities_tensor = [torch.FloatTensor(mod).unsqueeze(0) for mod in modalities_data]
        segmentation_tensor = torch.LongTensor(segmentation)
        
        # 应用变换
        if self.transform:
            modalities_tensor, segmentation_tensor = self.transform(
                modalities_tensor, segmentation_tensor
            )
        
        return modalities_tensor, segmentation_tensor
    
    def _convert_segmentation_labels(self, seg):
        """将分割标签转换为论文中的类别"""
        # BraTS标签: 0=背景, 1=坏死, 2=水肿, 4=增强肿瘤
        # 转换为: 0=背景, 1=坏死, 2=水肿, 3=增强肿瘤
        converted = np.zeros_like(seg)
        converted[seg == 1] = 1  # 坏死
        converted[seg == 2] = 2  # 水肿
        converted[seg == 4] = 3  # 增强肿瘤
        
        return converted