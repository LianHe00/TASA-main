# import os
# import numpy as np
# import open3d as o3d
# import json
# from tqdm import tqdm
# import argparse
# from dataset.AffordanceDataset import AffordanceDataset


# def process_mask_and_pointcloud_from_json(mask_json_path, dataset, visit_id, scan_id, desc_id, radius=0.1, num_points=8192):
#     """
#     根据mask_result.json中的original_indices，从原始点云中提取mask点，并进行分块处理。
#     """
#     # 读取mask_result.json
#     with open(mask_json_path, 'r') as f:
#         mask_data = json.load(f)
#     original_indices = mask_data['original_indices']
#     desc_id = mask_data.get('desc_id', '')
#     desc_text = mask_data.get('desc_text', '')
#     annot_id = mask_data.get('annot_id', '')

#     # 读取原始点云
#     laser_scan_path = dataset.get_data_asset_path(
#         split=split,
#         data_asset_identifier="laser_scan_5mm",
#         visit_id=visit_id
#     )
#     laser_scan = o3d.io.read_point_cloud(laser_scan_path)
#     # cropped_laser_scan: 分割后的点云, filtered_idx_list: 分割后的点云在原始点云中的编号, 如果需要把新点云i点在原始点云的编号，可以查看filtered_idx_list[i]
#     cropped_laser_scan, filtered_idx_list = dataset.get_cropped_laser_scan_and_id(visit_id, laser_scan)

#     cropped_indices = np.where(np.isin(filtered_idx_list, original_indices))[0]

    
#     # if not os.path.exists(raw_pcd_path):
#     #     print(f"原始点云文件不存在: {raw_pcd_path}")
#     #     return
#     # pcd = o3d.io.read_point_cloud(raw_pcd_path)
#     all_points = np.asarray(cropped_laser_scan.points)
#     all_colors = np.asarray(cropped_laser_scan.colors) if pcd.has_colors() else None

#     # 选出mask点
#     if len(cropped_indices) == 0:
#         print(f"{mask_json_path} 中 original_indices/cropped_indices 为空，跳过")
#         return
#     mask_points = all_points[cropped_indices]
#     mean_xyz = mask_points.mean(axis=0)

#     # 计算所有点到mask点均值的距离
#     distances = np.linalg.norm(all_points - mean_xyz, axis=1)
#     within_radius_mask = (distances <= radius)

#     # 保证采样点数不少于num_points
#     if within_radius_mask.sum() < num_points:
#         remaining_mask = ~within_radius_mask
#         remaining_distances = distances[remaining_mask]
#         remaining_indices = np.where(remaining_mask)[0]
#         num_to_add = num_points - within_radius_mask.sum()
#         if num_to_add > 0 and len(remaining_distances) > 0:
#             closest_indices = remaining_indices[np.argsort(remaining_distances)[:num_to_add]]
#             within_radius_mask[closest_indices] = True

#     # 如果点数多于num_points，取最近的num_points个
#     if within_radius_mask.sum() > num_points:
#         selected_indices = np.where(within_radius_mask)[0]
#         selected_distances = distances[selected_indices]
#         sorted_indices = np.argsort(selected_distances)[:num_points]
#         closest_indices = selected_indices[sorted_indices]
#         new_within_radius_mask = np.zeros_like(within_radius_mask, dtype=bool)
#         new_within_radius_mask[closest_indices] = True
#         within_radius_mask = new_within_radius_mask

#     # 生成分块后的点云
#     filtered_points = all_points[within_radius_mask]
#     filtered_point_cloud = o3d.geometry.PointCloud()
#     filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
#     if all_colors is not None:
#         filtered_colors = all_colors[within_radius_mask]
#         filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

#     # 保存结果
#     os.makedirs(save_dir, exist_ok=True)
#     # 保存json
#     result = {
#         "desc_id": desc_id,
#         "desc_text": desc_text,
#         "annot_id": annot_id,
#         "mask_point_num": len(filtered_point_cloud.points),
#         "mean_xyz": mean_xyz.tolist(),
#         "original_indices": np.where(within_radius_mask)[0].tolist()
#     }
#     json_path = os.path.join(save_dir, "mask_result.json")
#     with open(json_path, 'w') as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)
#     print(f"已保存json到 {json_path}")
#     # 保存ply
#     ply_path = os.path.join(save_dir, "filtered_point_cloud.ply")
#     o3d.io.write_point_cloud(ply_path, filtered_point_cloud)
#     print(f"已保存ply到 {ply_path}")
#     # 保存mask
#     filtered_mask = np.zeros(len(all_points), dtype=np.uint8)
#     filtered_mask[within_radius_mask] = 1
#     filtered_mask_path = os.path.join(save_dir, "filtered_mask.npy")
#     np.save(filtered_mask_path, filtered_mask)
#     print(f"已保存filtered_mask到 {filtered_mask_path}")
import os
import numpy as np
import open3d as o3d
import json
import sys
import argparse
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from dataset.AffordanceDataset import AffordanceDataset


def process_mask_and_pointcloud_from_json(mask_json_path, dataset, split, save_dir, visit_id, scan_id, desc_id, radius=0.1, num_points=8192):
    """
    根据mask_result.json中的original_indices，从原始点云中提取mask点，并进行分块处理。
    所有点云相关计算均在GPU上完成。
    """
    # 读取mask_result.json
    with open(mask_json_path, 'r') as f:
        mask_data = json.load(f)
    original_indices = mask_data['original_indices']
    desc_id = mask_data.get('desc_id', '')
    desc_text = ''
    annot_id = ''

    # 读取description
    descriptions = dataset.get_descriptions(visit_id)
    for desc in descriptions:
        if desc['desc_id'] == desc_id:
            desc_text = desc['description']
            annot_id = desc.get('annot_id')
            break
    if desc_text == '' or annot_id == '':
        print(f"error: desc_id: {desc_id} 在 descriptions 中不存在，跳过")
        exit()
    
    # 读取gt_mask
    gt_mask = dataset.get_grouped_annotation(visit_id, desc_id)
    gt_mask = torch.tensor(gt_mask)

    # 读取原始点云
    laser_scan_path = dataset.get_data_asset_path(
        split=split,
        data_asset_identifier="laser_scan_5mm",
        visit_id=visit_id
    )
    laser_scan = o3d.io.read_point_cloud(laser_scan_path)
    cropped_laser_scan, filtered_idx_list = dataset.get_cropped_laser_scan_and_id(visit_id, laser_scan)

    # 构建cropped mask indices
    cropped_indices = np.where(np.isin(filtered_idx_list, original_indices))[0]

    if len(cropped_indices) == 0:
        print(f"{mask_json_path} 中 original_indices/cropped_indices 为空，跳过")
        return
    
    # 构建pred_mask
    pred_mask = torch.zeros_like(gt_mask, dtype=torch.uint8)
    pred_mask[cropped_indices] = 1

    # 选择设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    all_points = torch.tensor(np.asarray(cropped_laser_scan.points), device=device, dtype=torch.float32)
    mask_points = all_points[cropped_indices]
    mean_xyz = mask_points.mean(dim=0)

    # 计算所有点到mask点均值的距离
    distances = torch.norm(all_points - mean_xyz, dim=1)
    within_radius_mask = (distances <= radius)  # torch.bool

    # 保证采样点数不少于num_points
    if within_radius_mask.sum().item() < num_points:
        remaining_mask = ~within_radius_mask
        remaining_distances = distances[remaining_mask]
        remaining_indices = torch.where(remaining_mask)[0]
        num_to_add = num_points - within_radius_mask.sum().item()
        if num_to_add > 0 and len(remaining_distances) > 0:
            closest_indices = remaining_indices[torch.argsort(remaining_distances)[:num_to_add]]
            within_radius_mask[closest_indices] = True

    # 如果点数多于num_points，取最近的num_points个
    if within_radius_mask.sum().item() > num_points:
        selected_indices = torch.where(within_radius_mask)[0]
        selected_distances = distances[selected_indices]
        sorted_indices = torch.argsort(selected_distances)[:num_points]
        closest_indices = selected_indices[sorted_indices]
        new_within_radius_mask = torch.zeros_like(within_radius_mask, dtype=torch.bool, device=device)
        new_within_radius_mask[closest_indices] = True
        within_radius_mask = new_within_radius_mask

    # 生成分块后的点云
    filtered_points = all_points[within_radius_mask].cpu().numpy()
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    if cropped_laser_scan.has_colors():
        all_colors = np.asarray(cropped_laser_scan.colors)
        filtered_colors = all_colors[within_radius_mask.cpu().numpy()]
        filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    # 保存结果
    os.makedirs(save_dir, exist_ok=True)
    # 保存json
    result = {
        "desc_id": desc_id,
        "desc_text": desc_text,
        "annot_id": annot_id,
        "mask_point_num": len(filtered_point_cloud.points),
        "mean_xyz": mean_xyz.cpu().numpy().tolist(),
        "original_indices": np.where(within_radius_mask.cpu().numpy())[0].tolist()
    }
    json_path = os.path.join(save_dir, "mask_result.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"已保存json到 {json_path}")

    # 保存分块后的点云
    ply_path = os.path.join(save_dir, "filtered_point_cloud.ply")
    o3d.io.write_point_cloud(ply_path, filtered_point_cloud)
    print(f"已保存ply到 {ply_path}")

    # 保存pred_mask（全局的）
    pred_mask_global_path = os.path.join(save_dir, "pred_mask_global.npy")
    np.save(pred_mask_global_path, pred_mask.cpu().numpy())
    print(f"已保存pred_mask_global到 {pred_mask_global_path}")

    # 保存pred_mask（分块后的）
    pred_mask_local_path = os.path.join(save_dir, "pred_mask_local.npy")
    np.save(pred_mask_local_path, pred_mask[within_radius_mask.cpu().numpy()])
    print(f"已保存pred_mask_local到 {pred_mask_local_path}")

    # 保存gt_mask（全局的）
    gt_mask_global_path = os.path.join(save_dir, "gt_mask_global.npy")
    np.save(gt_mask_global_path, gt_mask.cpu().numpy())
    print(f"已保存gt_mask_global到 {gt_mask_global_path}")

    # 保存gt_mask（分块后的）
    gt_mask_local_path = os.path.join(save_dir, "gt_mask_local.npy")
    np.save(gt_mask_local_path, gt_mask[within_radius_mask.cpu().numpy()])
    print(f"已保存gt_mask_local到 {gt_mask_local_path}")


def main():
    parser = argparse.ArgumentParser(description='根据mask_result.json对原始点云分块')
    parser.add_argument('--root_dir', type=str, default='data',help='数据根目录')
    parser.add_argument('--clip_dir', type=str, default='/data/helian/affseg/lift', help='clipwithaffordance根目录')
    parser.add_argument('--raw_data_dir', type=str, default='data/raw_data', help='原始点云根目录')
    parser.add_argument('--save_dir', type=str, default='data/processed_sam2_clipwithaffordance_manual_refine', help='结果保存根目录')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='只处理指定split')
    parser.add_argument('--radius', type=float, default=0.15, help='搜索半径')
    parser.add_argument('--num_points', type=int, default=65536, help='采样点数')
    args = parser.parse_args()

    # splits = ['train', 'val'] if args.split is None else [args.split]
    splits = ['val'] 
    for split in splits:
        # split_dir = os.path.join(args.clip_dir, split)
        split_dir = args.clip_dir
        if not os.path.exists(split_dir):
            print(f"目录不存在: {split_dir}")
            continue

        dataset = AffordanceDataset(
            root_dir=args.root_dir,
            split=args.split,
            use_processed_data=False,
            use_division=True
        )
        for visit_id in tqdm(os.listdir(split_dir), desc=f"处理{split}"):
            visit_path = os.path.join(split_dir, visit_id)
            if not os.path.isdir(visit_path):
                continue
            for scan_id in os.listdir(visit_path):
                scan_path = os.path.join(visit_path, scan_id)
                if not os.path.isdir(scan_path):
                    continue
                for desc_id in os.listdir(scan_path):
                    desc_path = os.path.join(scan_path, desc_id)
                    mask_json_path = os.path.join(desc_path, 'mask_result.json')
                    if not os.path.exists(mask_json_path):
                        continue
                    # 原始点云路径
                    raw_pcd_path = os.path.join(args.raw_data_dir, visit_id, f"{scan_id}.ply")
                    # 保存目录
                    save_dir = os.path.join(args.save_dir, split, visit_id, scan_id, desc_id)
                    try:
                        process_mask_and_pointcloud_from_json(
                            mask_json_path, dataset, split, save_dir, visit_id, scan_id, desc_id, radius=args.radius, num_points=args.num_points
                        )
                    except Exception as e:
                        print(f"处理 {mask_json_path} 时出错: {str(e)}")
                        continue
    print("——————————————全部处理完成——————————————")

if __name__ == "__main__":
    main()

# python dataset/preprocess_data_sam2.py --root_dir data --split val --radius 0.15 --num_points 65536
# /data/helian/affseg/lift

# python dataset/preprocess_data_sam2.py \
#  --root_dir data\
#  --split val\
#  --radius 0.15\
#  --clip_dir /data/helian/affseg/lift\
#  --save_dir /data/helian/affseg/processed_sam2_clipwithaffordance_manual_refine_new_65536\
#  --num_points 65536

