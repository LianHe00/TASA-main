import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
import random
import sys
from os.path import join
import glob
from dataset.data_parser_paths import data_asset_to_path
import open3d as o3d

import pdb

class AffordanceDataset(Dataset):
    def __init__(self, root_dir, split, use_processed_data=False, use_division=False, use_processed_data_3=False, 
    use_sam2=False, use_sam2_1=False, use_processed_final_train=False):
        """
        :param root_dir: 数据集根目录
        :param split: 数据集划分（'train' 或 'val'）
        :param use_processed_data: 是否使用预处理后的数据
        """
        self.root_dir = root_dir
        self.split = split
        self.use_processed_data = use_processed_data
        self.use_division = use_division
        self.use_processed_data_3 = use_processed_data_3
        self.use_sam2 = use_sam2
        self.use_sam2_1 = use_sam2_1
        self.use_processed_final_train = use_processed_final_train

        if self.use_sam2:
            # self.processed_dir = os.path.join(root_dir, 'processed_sam2_clipwithaffordance', split)
            # self.processed_dir = os.path.join(root_dir, 'processed_sam2_clipwithaffordance_manual_refine', split)
            self.processed_dir = '/data/helian/affseg/processed_sam2_clipwithaffordance_manual_refine_new/val'
            if not os.path.exists(self.processed_dir):
                raise ValueError(f"预处理数据目录 {self.processed_dir} 不存在，请先运行preprocess_data_sam2.py")
            self.data_items = []
            # 遍历所有visit_id/scan_id/desc_id目录，收集每个数据项
            for visit_id in os.listdir(self.processed_dir):
                visit_path = os.path.join(self.processed_dir, visit_id)
                if not os.path.isdir(visit_path):
                    continue
                for scan_id in os.listdir(visit_path):
                    scan_path = os.path.join(visit_path, scan_id)
                    if not os.path.isdir(scan_path):
                        continue
                    for desc_id in os.listdir(scan_path):
                        desc_path = os.path.join(scan_path, desc_id)
                        if not os.path.isdir(desc_path):
                            continue
                        # 检查六个文件是否都存在
                        files = [
                            "filtered_point_cloud.ply",
                            "gt_mask_global.npy",
                            "gt_mask_local.npy",
                            "mask_result.json",
                            "pred_mask_global.npy",
                            "pred_mask_local.npy"
                        ]
                        if all(os.path.exists(os.path.join(desc_path, f)) for f in files):
                            self.data_items.append({
                                "visit_id": visit_id,
                                "scan_id": scan_id,
                                "desc_id": desc_id,
                                "base_path": desc_path
                            })           

        if self.use_sam2_1:
            self.processed_dir = os.path.join(root_dir, 'processed_sam2_clipwithaffordance_1', split)
            if not os.path.exists(self.processed_dir):
                raise ValueError(f"预处理数据目录 {self.processed_dir} 不存在，请先运行preprocess_data_sam2_1.py")
            self.data_items = []
            # 遍历所有visit_id/scan_id/desc_id/image_id目录，收集每个数据项
            for visit_id in os.listdir(self.processed_dir):
                visit_path = os.path.join(self.processed_dir, visit_id)
                if not os.path.isdir(visit_path):
                    continue
                for scan_id in os.listdir(visit_path):
                    scan_path = os.path.join(visit_path, scan_id)
                    if not os.path.isdir(scan_path):
                        continue
                    for desc_id in os.listdir(scan_path):
                        desc_path = os.path.join(scan_path, desc_id)
                        if not os.path.isdir(desc_path):
                            continue
                        for image_id in os.listdir(desc_path):
                            image_path = os.path.join(desc_path, image_id)
                            if not os.path.isdir(image_path):
                                continue
                            # 检查六个文件是否都存在
                            files = [
                                "filtered_point_cloud.ply",
                                "gt_mask_global.npy",
                                "gt_mask_local.npy",
                                "mask_result.json",
                                "pred_mask_global.npy",
                                "pred_mask_local.npy"
                            ]
                            if all(os.path.exists(os.path.join(image_path, f)) for f in files):
                                self.data_items.append({
                                    "visit_id": visit_id,
                                    "scan_id": scan_id,
                                    "desc_id": desc_id,
                                    "image_id": image_id,
                                    "base_path": image_path
                                })
            # 注释：
            # data_items中的每一项为一个dict，包含该数据的visit_id、scan_id、desc_id、image_id和该image_id目录的base_path。
            # 后续getitem可直接用base_path拼接文件名读取数据。
        

        # 如果使用分割后的mask
        if self.use_division:
            self.processed_dir = os.path.join(root_dir, 'processed_data_segment_16385', split)
            if not os.path.exists(self.processed_dir):
                raise ValueError(f"预处理数据目录 {self.processed_dir} 不存在，请先运行preprocess_data2.py")
        
        # 如果使用预处理数据，检查是否存在
        if self.use_processed_data:
            self.processed_dir = os.path.join(root_dir, 'processed_data_sample_65536', split)
            if not os.path.exists(self.processed_dir):
                raise ValueError(f"预处理数据目录 {self.processed_dir} 不存在，请先运行preprocess_data.py")
            
            # 加载处理信息
            with open(os.path.join(self.processed_dir, 'process_info.json'), 'r') as f:
                self.process_info = json.load(f)
        
        if self.use_processed_data_3:
            self.processed_dir = os.path.join(root_dir, 'processed4', split)
            if not os.path.exists(self.processed_dir):
                raise ValueError(f"预处理数据目录 {self.processed_dir} 不存在，请先运行processed3.py")
        
        if self.use_processed_final_train:
            # self.processed_dir = os.path.join(root_dir, 'processed_65536', split)
            self.processed_dir = os.path.join('/data/helian/affseg/processed_data/division_8192', split)
            if not os.path.exists(self.processed_dir):
                raise ValueError(f"预处理数据目录 {self.processed_dir} 不存在")
            self.data_items = []
            # 遍历所有visit_id/scan_id/desc_id目录，收集每个数据项
            for visit_id in os.listdir(self.processed_dir):
                visit_path = os.path.join(self.processed_dir, visit_id)
                if not os.path.isdir(visit_path):
                    continue
                for desc_id in os.listdir(visit_path):
                    desc_path = os.path.join(visit_path, desc_id)
                    if not os.path.isdir(desc_path):
                        continue
                    # 检查六个文件是否都存在
                    files = [
                        "filtered_mask.npy",
                        "filtered_point_cloud.ply",
                        "gt_mask_global.npy",
                        "mask_result.json",
                    ]
                    if all(os.path.exists(os.path.join(desc_path, f)) for f in files):
                        with open(os.path.join(desc_path, "mask_result.json"), "r") as f:
                            mask_result = json.load(f)
                            description = mask_result['desc_text']
                            self.data_items.append({
                                "visit_id": visit_id,
                                "desc_id": desc_id,
                                "base_path": desc_path,
                                "description": desc_id
                            })  
        
        if not (self.use_sam2 or self.use_sam2_1 or self.use_processed_final_train):
            # 初始化数据集
            self.visit_ids = self.get_visit_id()
            
            # 构建数据索引：以description_id为主
            self.data_items = []
            for visit_id in self.visit_ids:
                descriptions = self.get_descriptions(visit_id)
                for desc in descriptions:
                    self.data_items.append({
                        'visit_id': visit_id,
                        'desc_id': desc['desc_id'],
                        'description': desc['description']
                    })

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        if self.use_sam2 or self.use_sam2_1:

            data_item = self.data_items[idx]
            base_path = data_item['base_path']
            # 读取点云
            pc_path = os.path.join(base_path, "filtered_point_cloud.ply")
            point_cloud = o3d.io.read_point_cloud(pc_path)
            points = torch.FloatTensor(np.asarray(point_cloud.points))
            colors = torch.FloatTensor(np.asarray(point_cloud.colors))
            # 读取mask和json
            gt_mask_global = torch.FloatTensor(np.load(os.path.join(base_path, "gt_mask_global.npy")))
            gt_mask_local = torch.FloatTensor(np.load(os.path.join(base_path, "gt_mask_local.npy")))
            pred_mask_global = torch.FloatTensor(np.load(os.path.join(base_path, "pred_mask_global.npy")))
            pred_mask_local = torch.FloatTensor(np.load(os.path.join(base_path, "pred_mask_local.npy")))
            with open(os.path.join(base_path, "mask_result.json"), "r") as f:
                mask_result = json.load(f)
            

            return {
                "c_pc_xyz": points, # 切块后的点云坐标
                "c_pc_feat": colors, # 切块后的点云颜色
                # "gt_mask_global": gt_mask_global, # gt的global mask
                "gt_mask_local": gt_mask_local, # gt的local mask
                # "pred_mask_global": pred_mask_global, # 预测的global mask
                "pred_mask_local": pred_mask_local,
                # "mask_result": mask_result,
                'c_text': mask_result['desc_text'],     # 文本描述
                "c_visit_id": data_item["visit_id"],
                "c_desc_id": data_item["desc_id"],
            }

        if self.use_processed_final_train:
            data_item = self.data_items[idx]
            base_path = data_item['base_path']
            description = data_item['description']
            pc_path = os.path.join(base_path, "filtered_point_cloud.ply")
            point_cloud = o3d.io.read_point_cloud(pc_path)
            points = torch.FloatTensor(np.asarray(point_cloud.points))
            colors = torch.FloatTensor(np.asarray(point_cloud.colors))
            mask = torch.FloatTensor(np.load(os.path.join(base_path, "filtered_mask.npy")))
            gt_mask_global = torch.FloatTensor(np.load(os.path.join(base_path, "gt_mask_global.npy")))
            gt_mask_local = torch.FloatTensor(np.load(os.path.join(base_path, "gt_mask_local.npy")))
            return {
                "x": mask,
                "c_pc_xyz": points,
                "c_pc_feat": colors,
                "gt_mask_global": gt_mask_global,
                "gt_mask_local": gt_mask_local,
                "c_text": description,
            }

        if self.use_processed_data_3:
            # 使用分割后的mask的新逻辑
            data_item = self.data_items[idx]
            visit_id = data_item['visit_id']
            desc_id = data_item['desc_id']
            description = data_item['description']
            
            # 加载原始点云
            laser_scan_path = self.get_data_asset_path(
                split=self.split,
                data_asset_identifier="laser_scan_5mm",
                visit_id=visit_id
            )
            laser_scan = o3d.io.read_point_cloud(laser_scan_path)
            original_pcd = self.get_cropped_laser_scan(visit_id, laser_scan)
            original_points = torch.FloatTensor(np.asarray(original_pcd.points))
            original_colors = torch.FloatTensor(np.asarray(original_pcd.colors))

            # 加载分割后的点云
            pc_path = os.path.join(self.processed_dir, f'{visit_id}/{desc_id}/filtered_point_cloud.ply')
            point_cloud = o3d.io.read_point_cloud(pc_path)
            
            # 加载分割后的mask
            mask_path = os.path.join(self.processed_dir, f'{visit_id}/{desc_id}/filtered_mask.npy')
            affordance_mask = np.load(mask_path)

            # 加载gt_mask
            gt_mask_path = os.path.join(self.processed_dir, f'{visit_id}/{desc_id}/gt_mask.npy')
            gt_mask = np.load(gt_mask_path)
            gt_mask = torch.FloatTensor(gt_mask)
            if gt_mask.dim() == 1:
                gt_mask = gt_mask.unsqueeze(1)

            # 加载original_indices
            mask_json_path = os.path.join(self.processed_dir, f'{visit_id}/{desc_id}/mask_result.json')
            with open(mask_json_path, 'r') as f:
                mask_result = json.load(f)
                original_indices = mask_result['original_indices']
                original_indices = torch.tensor(original_indices)
            
            # 转换为tensor
            points = torch.FloatTensor(np.asarray(point_cloud.points))
            colors = torch.FloatTensor(np.asarray(point_cloud.colors))
            mask = torch.FloatTensor(affordance_mask)
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)

            return {
                'pred_mask': mask,        # 预测标注
                'gt_mask': gt_mask,       # gt的mask
                'c_pc_xyz': points,        # 点云坐标
                'c_pc_feat': colors,       # 点云颜色
                'c_text': description,     # 文本描述
                'c_visit_id': visit_id,    # 用于追踪个
                'c_desc_id': desc_id,     # 用于追踪
                'c_original_pc_xyz': original_points,
                'c_original_pc_feat': original_colors,
                'original_indices': original_indices
            }
        
        elif self.use_division:
            # 使用分割后的mask的新逻辑
            data_item = self.data_items[idx]
            visit_id = data_item['visit_id']
            desc_id = data_item['desc_id']
            description = data_item['description']
            
            # 加载分割后的点云
            pc_path = os.path.join(self.processed_dir, f'{visit_id}/{desc_id}/filtered_point_cloud.ply')
            point_cloud = o3d.io.read_point_cloud(pc_path)
            
            # 加载分割后的mask
            mask_path = os.path.join(self.processed_dir, f'{visit_id}/{desc_id}/filtered_mask.npy')
            affordance_mask = np.load(mask_path)
            
            # 转换为tensor
            points = torch.FloatTensor(np.asarray(point_cloud.points))
            colors = torch.FloatTensor(np.asarray(point_cloud.colors))
            mask = torch.FloatTensor(affordance_mask)
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
        

            return {
                'x': mask,                 # affordance标注
                'c_pc_xyz': points,        # 点云坐标
                'c_pc_feat': colors,       # 点云颜色
                'c_text': description,     # 文本描述
                'c_visit_id': visit_id,    # 用于追踪个
                'c_desc_id': desc_id       # 用于追踪
            }
        else:
            # 原来的逻辑
            data_item = self.data_items[idx]
            visit_id = data_item['visit_id']
            desc_id = data_item['desc_id']
            description = data_item['description']
            if self.use_processed_data:
                # 加载预处理后的点云
                pc_path = os.path.join(self.processed_dir, 'point_clouds', f'{visit_id}.ply')
                point_cloud = o3d.io.read_point_cloud(pc_path)
                
                # 加载预处理后的mask
                mask_path = os.path.join(self.processed_dir, 'masks', f'{visit_id}_{desc_id}.npy')
                affordance_mask = np.load(mask_path)
            else:
                # 使用原始数据（保留原来的处理逻辑）
                laser_scan_path = self.get_data_asset_path(
                    split=self.split,
                    data_asset_identifier="laser_scan_5mm",
                    visit_id=visit_id
                )
                laser_scan = o3d.io.read_point_cloud(laser_scan_path)
                point_cloud = self.get_cropped_laser_scan(visit_id, laser_scan)
                affordance_mask = self.get_grouped_annotation(visit_id, desc_id)
            
            # 转换为tensor
            points = torch.FloatTensor(np.asarray(point_cloud.points))
            colors = torch.FloatTensor(np.asarray(point_cloud.colors))
            mask = torch.FloatTensor(affordance_mask)
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)

            return {
                'x': mask,                 # affordance标注
                'c_pc_xyz': points,        # 点云坐标
                'c_pc_feat': colors,       # 点云颜色
                'c_text': description,     # 文本描述
                'c_visit_id': visit_id,      # 用于追踪
                'c_desc_id': desc_id         # 用于追踪
            }
    # 获取所有visit_id,没有去重,没有排序
    def get_visit_id(self):
        with open(
            os.path.join(f"{self.root_dir}/raw_data/benchmark_file_lists/{self.split}_set.csv")
        ) as f:
            visit_video = f.readlines()[1:]

        visits = list()
        for line in visit_video:
            visit_id = line.strip("\n").split(",")[0]
            visits.append(visit_id)

        # 去重且保持顺序
        seen = set()
        unique_visits = []
        for vid in visits:
            if vid not in seen:
                unique_visits.append(vid)
                seen.add(vid)

        return unique_visits
    
    # 根据visit_id和video_id获取内部的数据路径
    def get_data_asset_path(self, split, data_asset_identifier, visit_id, video_id=None):
        assert (
            data_asset_identifier in data_asset_to_path
        ), f"Data asset identifier '{data_asset_identifier}' is not valid"

        data_path = data_asset_to_path[data_asset_identifier]

        if ("<video_id>" in data_path) and (video_id is None):
            assert (
                False
            ), f"video_id must be specified for the data asset identifier '{data_asset_identifier}'"

        ROOT = self.root_dir + '/raw_data/' + split
        visit_id = str(visit_id)

        data_path = data_path.replace("<data_dir>", ROOT).replace("<visit_id>", visit_id)

        if "<video_id>" in data_path:
            video_id = str(video_id)
            data_path = data_path.replace("<video_id>", video_id)

        return data_path
    
    # 根据visit_id和video_id获取每个video_id的rgb帧,帧和帧的文件路径对应
    def get_rgb_frames(self, visit_id, video_id, data_asset_identifier="hires_wide"):
        frame_mapping = {}
        
        if data_asset_identifier == "hires_wide":
            rgb_frames_path = self.get_data_asset_path(
                split=self.split, data_asset_identifier="hires_wide", visit_id=visit_id, video_id=video_id
            )

            frames = sorted(glob.glob(os.path.join(rgb_frames_path, "*.jpg")))
            if not frames:
                raise FileNotFoundError(f"No RGB frames found in {rgb_frames_path}")
            frame_timestamps = [
                os.path.basename(x).split(".jpg")[0].split("_")[1] for x in frames
            ]

        elif data_asset_identifier == "lowres_wide":
            rgb_frames_path = self.get_data_asset_path(
                data_asset_identifier="lowres_wide",
                visit_id=visit_id,
                video_id=video_id,
            )

            frames = sorted(glob.glob(os.path.join(rgb_frames_path, "*.png")))
            if not frames:
                raise FileNotFoundError(f"No RGB frames found in {rgb_frames_path}")
            frame_timestamps = [
                os.path.basename(x).split(".png")[0].split("_")[1] for x in frames
            ]
        else:
            raise ValueError(
                f"Unknown data_asset_identifier {data_asset_identifier} for RGB frames"
            )

        frame_mapping = {
            timestamp: frame for timestamp, frame in zip(frame_timestamps, frames)
        }

        return frame_mapping
    
    # 根据visit_id和video_id获取每个video_id的相机内参,返回帧时间戳和相机内参文件的路径
    def get_camera_intrinsics(self, visit_id, video_id, data_asset_identifier="hires_wide_intrinsics"):
        intrinsics_mapping = {}
        if data_asset_identifier == "hires_wide_intrinsics":
            intrinsics_path = self.get_data_asset_path(
                data_asset_identifier="hires_wide_intrinsics",
                split=self.split,
                visit_id=visit_id,
                video_id=video_id,
            )

        elif data_asset_identifier == "lowres_wide_intrinsics":
            intrinsics_path = self.get_data_asset_path(
                data_asset_identifier="lowres_wide_intrinsics",
                split=self.split,
                visit_id=visit_id,
                video_id=video_id,
            )

        else:
            raise ValueError(
                f"Unknown data_asset_identifier {data_asset_identifier} for camera intrinsics"
            )

        intrinsics = sorted(glob.glob(os.path.join(intrinsics_path, "*.pincam")))

        if not intrinsics:
            raise FileNotFoundError(f"No camera intrinsics found in {intrinsics_path}")

        intrinsics_timestamps = [
            os.path.basename(x).split(".pincam")[0].split("_")[1] for x in intrinsics
        ]

        intrinsics_mapping = {
            timestamp: cur_intrinsics
            for timestamp, cur_intrinsics in zip(intrinsics_timestamps, intrinsics)
        }

        return intrinsics_mapping
    
    # 根据visit_id获取每个visit_id的分割掩码crop_mask,返回crop_mask的值或者crop_mask的索引
    def get_crop_mask(self, visit_id, return_indices=False):
        crop_mask_path = self.get_data_asset_path(
            data_asset_identifier="crop_mask",
            split=self.split,
            visit_id=visit_id,
        )

        if not os.path.exists(crop_mask_path):
            raise FileNotFoundError(f"No crop mask found in {crop_mask_path}")
        
        crop_mask = np.load(crop_mask_path)
        
        if return_indices:
            return np.where(crop_mask)[0]
        else:
            return crop_mask

    # 根据visit_id及其对应的分割掩码crop_mask,返回分割后的点云
    def get_cropped_laser_scan(self, visit_id, laser_scan):
        filtered_idx_list = self.get_crop_mask(visit_id, return_indices=True)

        laser_scan_points = np.array(laser_scan.points)
        laser_scan_colors = np.array(laser_scan.colors)
        laser_scan_points = laser_scan_points[filtered_idx_list]
        laser_scan_colors = laser_scan_colors[filtered_idx_list]

        cropped_laser_scan = o3d.geometry.PointCloud()
        cropped_laser_scan.points = o3d.utility.Vector3dVector(laser_scan_points)
        cropped_laser_scan.colors = o3d.utility.Vector3dVector(laser_scan_colors)

        return cropped_laser_scan
        
    # 根据visit_id获取每个visit_id的全部描述
    def get_descriptions(self, visit_id):
        descriptions_path = self.get_data_asset_path(
            split=self.split, data_asset_identifier="descriptions", visit_id=visit_id
        )
        with open(descriptions_path, "r") as f:
            descriptions_data = json.load(f)["descriptions"]

        return descriptions_data
    
    # 根据visit_id获取每个visit_id的描述列表,返回值是 "desc_id" 和 description" 的键值对
    def get_descriptions_list(self, visit_id: str):
        descs = self.get_descriptions(visit_id)
        desc_ids = {desc["desc_id"]: desc["description"] for desc in descs}
        return desc_ids
    
    # 根据visit_id获取每个visit_id的全部功能注释,返回值是 "label" ， "indices" ，"label"
    def get_annotations(self, visit_id, group_excluded_points=True):
        annotations_path = self.get_data_asset_path(
            split=self.split, data_asset_identifier="annotations", visit_id=visit_id
        )

        annotations_data = None
        with open(annotations_path, "r") as f:
            annotations_data = json.load(f)["annotations"]

        if group_excluded_points:
            # group the excluded points into a single annotation instance
            exclude_indices_set = set()
            first_exclude_annotation = None
            filtered_annotation_data = []

            for annotation in annotations_data:
                if annotation["label"] == "exclude":
                    if first_exclude_annotation is None:
                        first_exclude_annotation = annotation
                    exclude_indices_set.update(annotation["indices"])
                else:
                    filtered_annotation_data.append(annotation)

            if first_exclude_annotation:
                first_exclude_annotation["indices"] = sorted(list(exclude_indices_set))
                filtered_annotation_data.append(first_exclude_annotation)

            annotations_data = filtered_annotation_data

        return annotations_data
    
    # 根据visit_id和desc_id获取desc_id对应的grouped_annotation,返回值是完整的mask
    def get_grouped_annotation(self, visit_id: str, desc_id: str, point_mapping=None) -> np.ndarray:
        """
        获取分组后的标注，支持降采样
        :param visit_id: 访问ID
        :param desc_id: 描述ID
        :param point_mapping: 点云降采样的映射关系
        :return: 标注mask
        """
        crop_mask = self.get_crop_mask(visit_id)
        full_mask = np.zeros(crop_mask.shape[0])

        descriptions = self.get_descriptions(visit_id)
        annots = self.get_annotations(visit_id)
        
        for desc in descriptions:
            if desc["desc_id"] == desc_id:
                annot_list = desc["annot_id"]
                break

        for annot in annots:
            if annot["annot_id"] in annot_list and annot["label"] != "exclude":
                idxs = np.asarray(annot["indices"])
                full_mask[idxs] = 1

        # 应用crop_mask
        full_mask = full_mask[crop_mask == 1]
        
        # 如果提供了point_mapping，将标注映射到降采样后的点云
        if point_mapping is not None:
            # 创建降采样后点云大小的新mask
            down_mask = np.zeros(len(np.unique(point_mapping)))
            # 对每个降采样后的点，如果它对应的原始点中有任何一个被标注，就标注这个点
            for i in range(len(down_mask)):
                # 找到映射到这个降采样点的所有原始点
                original_points = np.where(point_mapping == i)[0]
                # 如果这些原始点中有任何一个被标注，就标注这个降采样点
                if np.any(full_mask[original_points]):
                    down_mask[i] = 1
            return down_mask
            
        return full_mask
    
    # 根据visit_id获取其对应的点云数据
    def get_pointcloud_data(self, visit_id):
        laser_scan_path = self.get_data_asset_path(
                split=self.split,
                data_asset_identifier="laser_scan_5mm",
                visit_id=visit_id
            )
        laser_scan = o3d.io.read_point_cloud(laser_scan_path)
        return laser_scan

    # 根据visit_id及其对应的分割掩码crop_mask,返回分割后的点云,如果需要把新点云i点在原始点云的编号，可以查看filtered_idx_list[i]
    def get_cropped_laser_scan_and_id(self, visit_id, laser_scan):
        filtered_idx_list = self.get_crop_mask(visit_id, return_indices=True)

        laser_scan_points = np.array(laser_scan.points)
        laser_scan_colors = np.array(laser_scan.colors)
        laser_scan_points = laser_scan_points[filtered_idx_list]
        laser_scan_colors = laser_scan_colors[filtered_idx_list]

        cropped_laser_scan = o3d.geometry.PointCloud()
        cropped_laser_scan.points = o3d.utility.Vector3dVector(laser_scan_points)
        cropped_laser_scan.colors = o3d.utility.Vector3dVector(laser_scan_colors)

        return cropped_laser_scan, filtered_idx_list
    

    # 根据visit_id和desc_id、annot_id,获取这个annot_id对应的mask
    def get_single_annot_mask(self, visit_id: str, desc_id: str, annot_id: str) -> np.ndarray:
        """
        获取分组后的标注，支持降采样
        :param visit_id: 访问ID
        :param desc_id: 描述ID
        :param point_mapping: 点云降采样的映射关系
        :return: 标注mask
        """
        crop_mask = self.get_crop_mask(visit_id)  # 原始点云的crop掩码
        full_mask = np.zeros(crop_mask.shape[0], dtype=np.uint8)  # 原始点云长度

        # 读取annotations
        annots = self.get_annotations(visit_id, group_excluded_points=False)
        target_indices = None
        for annot in annots:
            if annot["annot_id"] == annot_id:
                target_indices = annot["indices"]
                break

        if target_indices is not None:
            full_mask[np.asarray(target_indices, dtype=int)] = 1

        # 应用crop_mask，得到分割后点云上的mask
        cropped_mask = full_mask[crop_mask == 1]
        return cropped_mask
        