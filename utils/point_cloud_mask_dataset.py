import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import json

class PointCloudMaskDataset(Dataset):
    """Point cloud mask refinement dataset.

    Data format:
    - point_cloud: point coords [N, 3]
    - initial_mask: initial mask [N] (0 or 1)
    - gt_mask: ground truth mask [N] (0 or 1)
    """

    def __init__(self, data_dir, split='train', max_points=65536, normalize=True):
        """
        Args:
            data_dir: path to data directory
            split: dataset split ('train' or 'val')
            max_points: max number of points
            normalize: whether to normalize point cloud
        """
        self.data_dir = data_dir
        self.split = split
        self.max_points = max_points
        self.normalize = normalize

        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} {split} samples from {data_dir}")
    
    def _load_samples(self):
        """Load all sample paths."""
        split_dir = os.path.join(self.data_dir, self.split)
        samples = []

        if not os.path.exists(split_dir):
            logger.error(f"Split directory {split_dir} does not exist!")
            return samples

        for scene_id in os.listdir(split_dir):
            scene_dir = os.path.join(split_dir, scene_id)
            if not os.path.isdir(scene_dir):
                continue

            for instance_id in os.listdir(scene_dir):
                instance_dir = os.path.join(scene_dir, instance_id)
                if not os.path.isdir(instance_dir):
                    continue

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
        """Load point cloud data (ASCII PLY)."""
        try:
            points = []
            with open(ply_path, 'r') as f:
                lines = f.readlines()

            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() == 'end_header':
                    data_start = i + 1
                    break

            for line in lines[data_start:]:
                if line.strip():
                    coords = [float(x) for x in line.strip().split()[:3]]
                    points.append(coords)

            points = np.array(points, dtype=np.float32)

            if self.normalize:
                center = np.mean(points, axis=0)
                points = points - center
                scale = np.max(np.linalg.norm(points, axis=1))
                points = points / scale
            
            return points
            
        except Exception as e:
            logger.error(f"Error loading point cloud from {ply_path}: {e}")
            return np.random.randn(1000, 3).astype(np.float32)

    def _load_mask(self, mask_path):
        """Load mask data."""
        try:
            mask = np.load(mask_path)
            return mask.astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading mask from {mask_path}: {e}")
            return np.random.randint(0, 2, 1000).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get single sample."""
        sample = self.samples[idx]

        point_cloud = self._load_point_cloud(sample['point_cloud_path'])
        initial_mask = self._load_mask(sample['initial_mask_path'])
        gt_mask = self._load_mask(sample['gt_mask_path'])

        num_points = min(len(point_cloud), len(initial_mask), len(gt_mask))
        point_cloud = point_cloud[:num_points]
        initial_mask = initial_mask[:num_points]
        gt_mask = gt_mask[:num_points]

        if num_points > self.max_points:
            indices = np.random.choice(num_points, self.max_points, replace=False)
            point_cloud = point_cloud[indices]
            initial_mask = initial_mask[indices]
            gt_mask = gt_mask[indices]

        point_cloud = torch.from_numpy(point_cloud).float()
        initial_mask = torch.from_numpy(initial_mask).float()
        gt_mask = torch.from_numpy(gt_mask).float()

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
    """Create point cloud mask dataloader."""
    
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
    """Custom collate for variable-length point clouds."""
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

    max_points = max(pc.shape[0] for pc in point_clouds)

    padded_point_clouds = []
    padded_initial_masks = []
    padded_gt_masks = []

    for pc, im, gm in zip(point_clouds, initial_masks, gt_masks):
        if pc.shape[0] < max_points:
            pad_size = max_points - pc.shape[0]
            pad_points = pc[-1:].repeat(pad_size, 1)
            pc = torch.cat([pc, pad_points], dim=0)

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