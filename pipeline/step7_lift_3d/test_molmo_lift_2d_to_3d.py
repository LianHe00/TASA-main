import os
import numpy as np
import open3d as o3d
import torch
import cv2
import json
import argparse
from scenefun3d_utils.data_parser import DataParser
from scenefun3d_utils.fusion_util import PointCloudToImageMapper
from tqdm import tqdm

# python test_molmo_lift_2d_to_3d.py --data_root data --split val

def process_one_molmo_mask(npz_path, data_root, split, visit_id, video_id, desc_id, all_results, output_dir=None):
    """处理单个Molmo掩码文件，将其从2D提升到3D，并将结果添加到all_results中"""
    
    # 1. 路径准备
    laser_scan_path = os.path.join(data_root, "raw_data", split, visit_id, f"{visit_id}_laser_scan.ply")
    traj_path = os.path.join(data_root, "raw_data", split, visit_id, video_id, "hires_poses.traj")

    # 2. 读取点云
    if not os.path.exists(laser_scan_path):
        print(f"❌ 点云文件不存在: {laser_scan_path}")
        return
    
    pcd = o3d.io.read_point_cloud(laser_scan_path)
    original_points = np.asarray(pcd.points)
    original_colors = np.asarray(pcd.colors)

    # 3. 获取裁剪索引
    parser = DataParser(os.path.join(data_root, "raw_data", split))
    crop_indices = parser.get_crop_mask(visit_id, return_indices=True)

    # 4. 裁剪点云
    points = original_points[crop_indices]
    colors = original_colors[crop_indices]

    # 5. 读取Molmo掩码数据
    npz = np.load(npz_path)
    masks = npz['masks']
    points_2d = npz['points'] if 'points' in npz else None
    
    print(f"处理掩码文件: {npz_path}")
    print(f"掩码数量: {len(masks)}")
    print(f"2D点数量: {len(points_2d) if points_2d is not None else 0}")

    # 6. 从文件名提取frame_id
    basename = os.path.basename(npz_path)
    
    # 支持多种格式：
    # 格式1: frame{frame_idx}_{video_id}_{frame_id}_crop_mask_data.npz
    # 格式2: {video_id}_{frame_id}_crop_mask_data.npz
    # 格式3: {video_id}_{frame_id}_mask_data.npz (新增支持)
    if basename.endswith('_crop_mask_data.npz'):
        if basename.startswith('frame'):
            # 格式1: frame{frame_idx}_{video_id}_{frame_id}_crop_mask_data.npz
            name_part = basename.replace('frame', '').replace('_crop_mask_data.npz', '')
            parts = name_part.split('_')
            if len(parts) >= 3:
                frame_id = parts[-1]  # 最后一个部分是frame_id
            else:
                print(f"❌ 无法从文件名解析frame_id: {basename}")
                return
        else:
            # 格式2: {video_id}_{frame_id}_crop_mask_data.npz
            name_part = basename.replace('_crop_mask_data.npz', '')
            parts = name_part.split('_')
            if len(parts) >= 2:
                frame_id = parts[-1]  # 最后一个部分是frame_id
            else:
                print(f"❌ 无法从文件名解析frame_id: {basename}")
                return
    elif basename.endswith('_mask_data.npz'):
        # 格式3: {video_id}_{frame_id}_mask_data.npz
        name_part = basename.replace('_mask_data.npz', '')
        parts = name_part.split('_')
        if len(parts) >= 2:
            frame_id = parts[-1]  # 最后一个部分是frame_id
        else:
            print(f"❌ 无法从文件名解析frame_id: {basename}")
            return
    else:
        print(f"❌ 文件名格式不正确: {basename}")
        print(f"   支持格式: *_crop_mask_data.npz 或 *_mask_data.npz")
        return

    # 7. 读取深度、内参、外参
    depth_dir = os.path.join(data_root, "raw_data", split, visit_id, video_id, "hires_depth")
    depth_path = os.path.join(depth_dir, f"{video_id}_{frame_id}.png")
    if not os.path.exists(depth_path):
        print(f"❌ 深度文件不存在: {depth_path}")
        return
    
    depth = parser.read_depth_frame(depth_path)
    h, w = depth.shape
    
    # 内参
    intrinsics_dir = os.path.join(data_root, "raw_data", split, visit_id, video_id, "hires_wide_intrinsics")
    intrinsics_files = [f for f in os.listdir(intrinsics_dir) if f.endswith('.pincam')]
    intrinsic_path = None
    for f in intrinsics_files:
        if frame_id in f:
            intrinsic_path = os.path.join(intrinsics_dir, f)
            break
    
    if intrinsic_path is None:
        print(f"❌ 相机内参文件不存在: {intrinsics_dir} (frame_id={frame_id})")
        return
    
    intrinsic = parser.read_camera_intrinsics(intrinsic_path, format="matrix")
    
    # 轨迹
    poses = {}
    with open(traj_path) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 7:
                ts = tokens[0]
                angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
                r_w_to_p = cv2.Rodrigues(np.asarray(angle_axis))[0]
                t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
                extrinsics = np.eye(4, 4)
                extrinsics[:3, :3] = r_w_to_p
                extrinsics[:3, -1] = t_w_to_p
                Rt = np.linalg.inv(extrinsics)
                poses[ts] = Rt
    
    pose = parser.get_nearest_pose(frame_id, poses)
    if pose is None:
        print(f"❌ 找不到合适的相机位姿: {frame_id}")
        return

    # 8. 处理每个掩码
    if torch.cuda.is_available():
        proc_pcd = torch.tensor(points).cuda()
        device = "cuda"
    else:
        proc_pcd = torch.tensor(points)
        device = "cpu"
    
    mask_threshold = 0.5
    
    # 为每个掩码单独处理
    for mask_idx, mask in enumerate(masks):
        print(f"处理 mask {mask_idx}")
        
        # 确保掩码尺寸正确
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 创建whole_mask
        whole_mask = np.ones(depth.shape)
        
        # 将掩码转换为torch tensor
        mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(torch.float)
        whole_mask_tensor = torch.tensor(whole_mask).unsqueeze(0).unsqueeze(0).to(torch.float)
        if torch.cuda.is_available():
            mask_tensor = mask_tensor.cuda()
            whole_mask_tensor = whole_mask_tensor.cuda()
        
        # 使用compute_multi_masked_mapping进行投影
        mapper = PointCloudToImageMapper((w, h))
        mapping_fo = mapper.compute_multi_masked_mapping(
            pose,
            proc_pcd,
            torch.stack([mask_tensor.squeeze(), whole_mask_tensor.squeeze()], dim=0),
            depth,
            intrinsic,
            device
        )
        
        # 获取有效映射
        valid_f = mapping_fo[0, :, -1] == 1
        valid_count = np.count_nonzero(valid_f)
        
        if valid_count > 0:
            # 获取有效点的坐标
            valid_fy = mapping_fo[0, valid_f, 0].astype(int)
            valid_fx = mapping_fo[0, valid_f, 1].astype(int)
            valid_point_indices = np.where(valid_f)[0]
            
            # 应用掩码阈值
            mask_values = mask[valid_fy, valid_fx]
            mask_points = mask_values > mask_threshold
            
            if np.sum(mask_points) > 0:
                mask_point_indices = valid_point_indices[mask_points]
                
                # 转换为排序后的列表
                cropped_mask_indices = sorted(mask_point_indices)
                original_mask_indices = [crop_indices[idx] for idx in cropped_mask_indices]
                
                # 提取掩码区域的点云
                mask_points_3d = original_points[original_mask_indices]
                mask_colors_3d = original_colors[original_mask_indices]
                mean_xyz = np.mean(mask_points_3d, axis=0).tolist()
                
                # 创建结果数据
                result_data = {
                    "desc_id": desc_id,
                    "desc_text": f"Molmo mask segmentation for {desc_id} (mask {mask_idx})",
                    "annot_id": [f"molmo_mask_{mask_idx}"],
                    "mask_point_num": int(len(original_mask_indices)),
                    "mean_xyz": [float(x) for x in mean_xyz],
                    "original_indices": [int(idx) for idx in original_mask_indices],
                    "frame_id": frame_id,
                    "mask_idx": mask_idx,
                    "source": "molmo_sam",
                    "input_file": npz_path,
                    "visit_id": visit_id,
                    "video_id": video_id
                }
                
                # 如果有2D点信息，也保存
                if points_2d is not None and len(points_2d) > mask_idx:
                    result_data["points_2d"] = points_2d[mask_idx].tolist() if hasattr(points_2d[mask_idx], 'tolist') else points_2d[mask_idx]
                
                # 添加到总结果中
                all_results.append(result_data)
                
                # 实时保存到文件（如果指定了输出目录）
                if output_dir is not None:
                    temp_output_path = os.path.join(output_dir, f"temp_{visit_id}_{video_id}_{desc_id}.json")
                    with open(temp_output_path, 'w') as f:
                        json.dump(all_results, f, indent=2, ensure_ascii=False)
                
                print(f"✅ 已处理 mask {mask_idx}")
                print(f"   点数: {len(original_mask_indices):,}")
            else:
                print(f"⚠️  mask {mask_idx} 没有点满足阈值 {mask_threshold}")
        else:
            print(f"❌ mask {mask_idx} 没有有效的点云投影")
    
    print(f"完成处理 {npz_path}")


def process_desc_directory(desc_dir, data_root, split, visit_id, video_id, desc_id, all_results, output_dir=None):
    """处理一个desc目录下的所有npz文件"""
    
    if not os.path.exists(desc_dir):
        print(f"警告: desc目录不存在: {desc_dir}")
        return
    
    # 查找所有掩码文件
    npz_files = [f for f in os.listdir(desc_dir) if f.endswith('.npz')]
    
    if len(npz_files) == 0:
        print(f"警告: 在 {desc_dir} 中未找到npz文件")
        return
    
    print(f"处理 {desc_id}: 找到 {len(npz_files)} 个npz文件")
    
    # 处理每个npz文件
    for npz_file in npz_files:
        npz_path = os.path.join(desc_dir, npz_file)
        process_one_molmo_mask(npz_path, data_root, split, visit_id, video_id, desc_id, all_results, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Molmo 2D到3D掩码提升工具')
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录路径')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'], help='数据集分割')
    parser.add_argument('--visit_id', type=str, default=None, help='特定的visit_id，如果不指定则处理所有')
    parser.add_argument('--video_id', type=str, default=None, help='特定的video_id，如果不指定则处理所有')
    parser.add_argument('--desc_id', type=str, default=None, help='特定的desc_id，如果不指定则处理所有')
    parser.add_argument('--real_time_save', action='store_true', help='启用实时保存功能')
    args = parser.parse_args()
    
    data_root = args.data_root
    split = args.split
    molmo_root = '/data/helian/affseg/molmo_merge'
    
    # 指定的visit_id列表
    target_visit_ids = ['421254', '421393', '423070', '434897', '435324', '437157']
    
    # 获取需要处理的visit_id
    if args.visit_id:
        visit_ids = [args.visit_id]
    else:
        # 处理所有visit_id
        visit_ids = [d for d in os.listdir(molmo_root) 
                    if os.path.isdir(os.path.join(molmo_root, d))]
        visit_ids = sorted(visit_ids)
    
    print(f"将处理 {len(visit_ids)} 个visit_id: {visit_ids}")
    
    # 存储所有结果的列表
    all_results = []
    
    # 处理每个visit_id
    for visit_id in tqdm(visit_ids, desc='visit_id'):
        visit_dir = os.path.join(molmo_root, visit_id)
        if not os.path.exists(visit_dir):
            print(f"警告: visit目录不存在: {visit_dir}")
            continue
        
        # 获取该visit下的所有video_id
        if args.video_id:
            video_ids = [args.video_id]
        else:
            video_ids = [d for d in os.listdir(visit_dir) 
                        if os.path.isdir(os.path.join(visit_dir, d))]
        
        print(f"Visit {visit_id} 包含 {len(video_ids)} 个视频: {video_ids}")
        
        # 处理每个video_id
        for video_id in video_ids:
            video_dir = os.path.join(visit_dir, video_id)
            if not os.path.exists(video_dir):
                print(f"警告: video目录不存在: {video_dir}")
                continue
            
            # 获取该video下的所有desc_id
            if args.desc_id:
                desc_ids = [args.desc_id]
            else:
                desc_ids = [d for d in os.listdir(video_dir) 
                           if os.path.isdir(os.path.join(video_dir, d))]
            
            # 处理每个desc_id
            for desc_id in desc_ids:
                desc_dir = os.path.join(video_dir, desc_id)
                process_desc_directory(desc_dir, data_root, split, visit_id, video_id, desc_id, all_results, output_dir if real_time_save else None)
    
    # 设置输出目录
    output_dir = '/data/helian/affseg/lift'
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据参数决定是否启用实时保存
    real_time_save = args.real_time_save
    if real_time_save:
        print("✅ 已启用实时保存功能，处理过程中会实时保存结果")
    else:
        print("⚠️  未启用实时保存功能，结果将在处理完成后统一保存")
    
    # 保存所有结果到一个json文件
    if all_results:
        
        # 生成输出文件名
        if args.visit_id and args.video_id and args.desc_id:
            output_filename = f"lift_results_{args.visit_id}_{args.video_id}_{args.desc_id}.json"
        elif args.visit_id:
            output_filename = f"lift_results_{args.visit_id}.json"
        else:
            output_filename = f"lift_results_all.json"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # 保存结果
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 所有处理完成！共处理了 {len(all_results)} 个掩码")
        print(f"✅ 结果已保存到: {output_path}")
    else:
        print("❌ 没有处理到任何有效的掩码数据")
    
    print("所有处理完成！")


if __name__ == '__main__':
    main() 