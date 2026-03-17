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
        self.root_dir = root_dir
        self.split = split
        self.use_processed_data = use_processed_data
        self.use_division = use_division
        self.use_processed_data_3 = use_processed_data_3
        self.use_sam2 = use_sam2
        self.use_sam2_1 = use_sam2_1
        self.use_processed_final_train = use_processed_final_train

        if self.use_sam2:
            self.processed_dir = os.path.join(root_dir, 'processed_sam2', split)
            if not os.path.exists(self.processed_dir):
                raise ValueError(f"Processed dir not found: {self.processed_dir}, run preprocess_data_sam2.py first")
            self.data_items = []
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
                raise ValueError(f"Processed dir not found: {self.processed_dir}")
            self.data_items = []
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
        if self.use_division:
            self.processed_dir = os.path.join(root_dir, 'processed_data_segment_16385', split)
            if not os.path.exists(self.processed_dir):
                raise ValueError(f"Processed dir not found: {self.processed_dir}")
        if self.use_processed_data:
            self.processed_dir = os.path.join(root_dir, 'processed_data_sample_65536', split)
            if not os.path.exists(self.processed_dir):
                raise ValueError(f"Processed dir not found: {self.processed_dir}")
            with open(os.path.join(self.processed_dir, 'process_info.json'), 'r') as f:
                self.process_info = json.load(f)
        
        if self.use_processed_data_3:
            self.processed_dir = os.path.join(root_dir, 'processed4', split)
            if not os.path.exists(self.processed_dir):
                raise ValueError(f"Processed dir not found: {self.processed_dir}")
        if self.use_processed_final_train:
            self.processed_dir = os.path.join(root_dir, 'processed_data', 'division_8192', split)
            if not os.path.exists(self.processed_dir):
                raise ValueError(f"Processed dir not found: {self.processed_dir}")
            self.data_items = []
            for visit_id in os.listdir(self.processed_dir):
                visit_path = os.path.join(self.processed_dir, visit_id)
                if not os.path.isdir(visit_path):
                    continue
                for desc_id in os.listdir(visit_path):
                    desc_path = os.path.join(visit_path, desc_id)
                    if not os.path.isdir(desc_path):
                        continue
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
            self.visit_ids = self.get_visit_id()
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
            pc_path = os.path.join(base_path, "filtered_point_cloud.ply")
            point_cloud = o3d.io.read_point_cloud(pc_path)
            points = torch.FloatTensor(np.asarray(point_cloud.points))
            colors = torch.FloatTensor(np.asarray(point_cloud.colors))
            gt_mask_global = torch.FloatTensor(np.load(os.path.join(base_path, "gt_mask_global.npy")))
            gt_mask_local = torch.FloatTensor(np.load(os.path.join(base_path, "gt_mask_local.npy")))
            pred_mask_global = torch.FloatTensor(np.load(os.path.join(base_path, "pred_mask_global.npy")))
            pred_mask_local = torch.FloatTensor(np.load(os.path.join(base_path, "pred_mask_local.npy")))
            with open(os.path.join(base_path, "mask_result.json"), "r") as f:
                mask_result = json.load(f)
            

            return {
                "c_pc_xyz": points,
                "c_pc_feat": colors,
                "gt_mask_local": gt_mask_local,
                "pred_mask_local": pred_mask_local,
                'c_text': mask_result['desc_text'],
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
            data_item = self.data_items[idx]
            visit_id = data_item['visit_id']
            desc_id = data_item['desc_id']
            description = data_item['description']
            
            laser_scan_path = self.get_data_asset_path(
                split=self.split,
                data_asset_identifier="laser_scan_5mm",
                visit_id=visit_id
            )
            laser_scan = o3d.io.read_point_cloud(laser_scan_path)
            original_pcd = self.get_cropped_laser_scan(visit_id, laser_scan)
            original_points = torch.FloatTensor(np.asarray(original_pcd.points))
            original_colors = torch.FloatTensor(np.asarray(original_pcd.colors))
            pc_path = os.path.join(self.processed_dir, f'{visit_id}/{desc_id}/filtered_point_cloud.ply')
            point_cloud = o3d.io.read_point_cloud(pc_path)
            mask_path = os.path.join(self.processed_dir, f'{visit_id}/{desc_id}/filtered_mask.npy')
            affordance_mask = np.load(mask_path)
            gt_mask_path = os.path.join(self.processed_dir, f'{visit_id}/{desc_id}/gt_mask.npy')
            gt_mask = np.load(gt_mask_path)
            gt_mask = torch.FloatTensor(gt_mask)
            if gt_mask.dim() == 1:
                gt_mask = gt_mask.unsqueeze(1)
            mask_json_path = os.path.join(self.processed_dir, f'{visit_id}/{desc_id}/mask_result.json')
            with open(mask_json_path, 'r') as f:
                mask_result = json.load(f)
                original_indices = mask_result['original_indices']
                original_indices = torch.tensor(original_indices)
            points = torch.FloatTensor(np.asarray(point_cloud.points))
            colors = torch.FloatTensor(np.asarray(point_cloud.colors))
            mask = torch.FloatTensor(affordance_mask)
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)

            return {
                'pred_mask': mask,
                'gt_mask': gt_mask,
                'c_pc_xyz': points,
                'c_pc_feat': colors,
                'c_text': description,
                'c_visit_id': visit_id,
                'c_desc_id': desc_id,
                'c_original_pc_xyz': original_points,
                'c_original_pc_feat': original_colors,
                'original_indices': original_indices
            }
        
        elif self.use_division:
            data_item = self.data_items[idx]
            visit_id = data_item['visit_id']
            desc_id = data_item['desc_id']
            description = data_item['description']
            pc_path = os.path.join(self.processed_dir, f'{visit_id}/{desc_id}/filtered_point_cloud.ply')
            point_cloud = o3d.io.read_point_cloud(pc_path)
            mask_path = os.path.join(self.processed_dir, f'{visit_id}/{desc_id}/filtered_mask.npy')
            affordance_mask = np.load(mask_path)
            points = torch.FloatTensor(np.asarray(point_cloud.points))
            colors = torch.FloatTensor(np.asarray(point_cloud.colors))
            mask = torch.FloatTensor(affordance_mask)
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
        

            return {
                'x': mask,
                'c_pc_xyz': points,
                'c_pc_feat': colors,
                'c_text': description,
                'c_visit_id': visit_id,
                'c_desc_id': desc_id
            }
        else:
            data_item = self.data_items[idx]
            visit_id = data_item['visit_id']
            desc_id = data_item['desc_id']
            description = data_item['description']
            if self.use_processed_data:
                pc_path = os.path.join(self.processed_dir, 'point_clouds', f'{visit_id}.ply')
                point_cloud = o3d.io.read_point_cloud(pc_path)
                mask_path = os.path.join(self.processed_dir, 'masks', f'{visit_id}_{desc_id}.npy')
                affordance_mask = np.load(mask_path)
            else:
                laser_scan_path = self.get_data_asset_path(
                    split=self.split,
                    data_asset_identifier="laser_scan_5mm",
                    visit_id=visit_id
                )
                laser_scan = o3d.io.read_point_cloud(laser_scan_path)
                point_cloud = self.get_cropped_laser_scan(visit_id, laser_scan)
                affordance_mask = self.get_grouped_annotation(visit_id, desc_id)
            
            points = torch.FloatTensor(np.asarray(point_cloud.points))
            colors = torch.FloatTensor(np.asarray(point_cloud.colors))
            mask = torch.FloatTensor(affordance_mask)
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
            return {
                'x': mask,
                'c_pc_xyz': points,
                'c_pc_feat': colors,
                'c_text': description,
                'c_visit_id': visit_id,
                'c_desc_id': desc_id
            }
    def get_visit_id(self):
        with open(
            os.path.join(f"{self.root_dir}/raw_data/benchmark_file_lists/{self.split}_set.csv")
        ) as f:
            visit_video = f.readlines()[1:]

        visits = list()
        for line in visit_video:
            visit_id = line.strip("\n").split(",")[0]
            visits.append(visit_id)
        seen = set()
        unique_visits = []
        for vid in visits:
            if vid not in seen:
                unique_visits.append(vid)
                seen.add(vid)

        return unique_visits
    
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
        
    def get_descriptions(self, visit_id):
        descriptions_path = self.get_data_asset_path(
            split=self.split, data_asset_identifier="descriptions", visit_id=visit_id
        )
        with open(descriptions_path, "r") as f:
            descriptions_data = json.load(f)["descriptions"]

        return descriptions_data

    def get_descriptions_list(self, visit_id: str):
        descs = self.get_descriptions(visit_id)
        desc_ids = {desc["desc_id"]: desc["description"] for desc in descs}
        return desc_ids

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
    
    def get_grouped_annotation(self, visit_id: str, desc_id: str, point_mapping=None) -> np.ndarray:
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
        full_mask = full_mask[crop_mask == 1]
        if point_mapping is not None:
            down_mask = np.zeros(len(np.unique(point_mapping)))
            for i in range(len(down_mask)):
                original_points = np.where(point_mapping == i)[0]
                if np.any(full_mask[original_points]):
                    down_mask[i] = 1
            return down_mask
            
        return full_mask

    def get_pointcloud_data(self, visit_id):
        laser_scan_path = self.get_data_asset_path(
                split=self.split,
                data_asset_identifier="laser_scan_5mm",
                visit_id=visit_id
            )
        laser_scan = o3d.io.read_point_cloud(laser_scan_path)
        return laser_scan

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
    

    def get_single_annot_mask(self, visit_id: str, desc_id: str, annot_id: str) -> np.ndarray:
        crop_mask = self.get_crop_mask(visit_id)
        full_mask = np.zeros(crop_mask.shape[0], dtype=np.uint8)
        annots = self.get_annotations(visit_id, group_excluded_points=False)
        target_indices = None
        for annot in annots:
            if annot["annot_id"] == annot_id:
                target_indices = annot["indices"]
                break

        if target_indices is not None:
            full_mask[np.asarray(target_indices, dtype=int)] = 1
        cropped_mask = full_mask[crop_mask == 1]
        return cropped_mask
        