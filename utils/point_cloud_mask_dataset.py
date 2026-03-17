import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import json

class PointCloudMaskDataset(Dataset):
    """点云mask润色数据集
    
    数据格式：
    - point_cloud: 点云坐标 [N, 3]
    - initial_mask: 初始mask [N] (0或1)
    - gt_mask: 真实mask [N] (0或1)
    """
    
    def __init__(self, data_dir, split='train', max_points=65536, normalize=True):
        """
        Args:
            data_dir: 数据目录路径
            split: 数据集分割 ('train' 或 'val')
            max_points: 最大点数
            normalize: 是否归一化点云
        """
        self.data_dir = data_dir
        self.split = split
        self.max_points = max_points
        self.normalize = normalize
        
        # 获取所有样本路径
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} {split} samples from {data_dir}")
    
    def _load_samples(self):
        """加载所有样本路径"""
        split_dir = os.path.join(self.data_dir, self.split)
        samples = []
        
        if not os.path.exists(split_dir):
            logger.error(f"Split directory {split_dir} does not exist!")
            return samples
        
        # 遍历所有场景目录
        for scene_id in os.listdir(split_dir):
            scene_dir = os.path.join(split_dir, scene_id)
            if not os.path.isdir(scene_dir):
                continue
                
            # 遍历场景中的所有实例
            for instance_id in os.listdir(scene_dir):
                instance_dir = os.path.join(scene_dir, instance_id)
                if not os.path.isdir(instance_dir):
                    continue
                
                # 检查必要文件是否存在
                point_cloud_path = os.path.join(instance_dir, 'filtered_point_cloud.ply')
                initial_mask_path = os.path.join(instance_dir, 'filtered_mask.npy')
                gt_mask_path = os.path.join(instance_dir, 'gt_mask.npy')
                
                if all(os.path.exists(p) for p in [point_cloud_path, initial_mask_path, gt_mask_path]):
                    samples.append({
                        'point_cloud_path': point_cloud_path,
                        'initial_mask_path': initial_mask_path,
                        'gt_mask_path': gt_mask_path,
                        'scene_id': scene_id,
                        'instance_id': instance_id
                    })
        
        return samples
    
    def _load_point_cloud(self, ply_path):
        """加载点云数据"""
        try:
            # 简单的PLY文件读取（假设是ASCII格式）
            points = []
            with open(ply_path, 'r') as f:
                lines = f.readlines()
                
            # 找到数据开始位置
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() == 'end_header':
                    data_start = i + 1
                    break
            
            # 读取点云数据
            for line in lines[data_start:]:
                if line.strip():
                    coords = [float(x) for x in line.strip().split()[:3]]
                    points.append(coords)
            
            points = np.array(points, dtype=np.float32)
            
            # 归一化点云
            if self.normalize:
                center = np.mean(points, axis=0)
                points = points - center
                scale = np.max(np.linalg.norm(points, axis=1))
                points = points / scale
            
            return points
            
        except Exception as e:
            logger.error(f"Error loading point cloud from {ply_path}: {e}")
            # 返回随机点云作为fallback
            return np.random.randn(1000, 3).astype(np.float32)
    
    def _load_mask(self, mask_path):
        """加载mask数据"""
        try:
            mask = np.load(mask_path)
            return mask.astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading mask from {mask_path}: {e}")
            # 返回随机mask作为fallback
            return np.random.randint(0, 2, 1000).astype(np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.samples[idx]
        
        # 加载数据
        point_cloud = self._load_point_cloud(sample['point_cloud_path'])
        initial_mask = self._load_mask(sample['initial_mask_path'])
        gt_mask = self._load_mask(sample['gt_mask_path'])
        
        # 确保点云和mask长度一致
        num_points = min(len(point_cloud), len(initial_mask), len(gt_mask))
        point_cloud = point_cloud[:num_points]
        initial_mask = initial_mask[:num_points]
        gt_mask = gt_mask[:num_points]
        
        # 如果点数超过限制，随机采样
        if num_points > self.max_points:
            indices = np.random.choice(num_points, self.max_points, replace=False)
            point_cloud = point_cloud[indices]
            initial_mask = initial_mask[indices]
            gt_mask = gt_mask[indices]
        
        # 转换为tensor
        point_cloud = torch.from_numpy(point_cloud).float()
        initial_mask = torch.from_numpy(initial_mask).float()
        gt_mask = torch.from_numpy(gt_mask).float()
        
        # 确保mask是二值化的（0或1）
        initial_mask = (initial_mask > 0.5).float()
        gt_mask = (gt_mask > 0.5).float()
        
        return {
            'point_cloud': point_cloud,  # [N, 3]
            'initial_mask': initial_mask,  # [N]
            'gt_mask': gt_mask,  # [N]
            'scene_id': sample['scene_id'],
            'instance_id': sample['instance_id']
        }

def create_point_cloud_mask_dataloader(data_dir, split='train', batch_size=8, 
                                     num_workers=4, max_points=65536, normalize=True):
    """创建点云mask数据加载器"""
    
    dataset = PointCloudMaskDataset(
        data_dir=data_dir,
        split=split,
        max_points=max_points,
        normalize=normalize
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader

def collate_point_cloud_mask(batch):
    """自定义collate函数，处理不同长度的点云"""
    point_clouds = []
    initial_masks = []
    gt_masks = []
    scene_ids = []
    instance_ids = []
    
    for item in batch:
        point_clouds.append(item['point_cloud'])
        initial_masks.append(item['initial_mask'])
        gt_masks.append(item['gt_mask'])
        scene_ids.append(item['scene_id'])
        instance_ids.append(item['instance_id'])
    
    # 找到最大点数
    max_points = max(pc.shape[0] for pc in point_clouds)
    
    # 填充到相同长度
    padded_point_clouds = []
    padded_initial_masks = []
    padded_gt_masks = []
    
    for pc, im, gm in zip(point_clouds, initial_masks, gt_masks):
        # 填充点云
        if pc.shape[0] < max_points:
            # 重复最后一个点来填充
            pad_size = max_points - pc.shape[0]
            pad_points = pc[-1:].repeat(pad_size, 1)
            pc = torch.cat([pc, pad_points], dim=0)
        
        # 填充mask
        if im.shape[0] < max_points:
            pad_size = max_points - im.shape[0]
            im = torch.cat([im, torch.zeros(pad_size)], dim=0)
        
        if gm.shape[0] < max_points:
            pad_size = max_points - gm.shape[0]
            gm = torch.cat([gm, torch.zeros(pad_size)], dim=0)
        
        padded_point_clouds.append(pc)
        padded_initial_masks.append(im)
        padded_gt_masks.append(gm)
    
    return {
        'point_cloud': torch.stack(padded_point_clouds),  # [B, N, 3]
        'initial_mask': torch.stack(padded_initial_masks),  # [B, N]
        'gt_mask': torch.stack(padded_gt_masks),  # [B, N]
        'scene_ids': scene_ids,
        'instance_ids': instance_ids
    } 