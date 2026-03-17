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
    with open(mask_json_path, 'r') as f:
        mask_data = json.load(f)
    original_indices = mask_data['original_indices']
    desc_id = mask_data.get('desc_id', '')
    desc_text = ''
    annot_id = ''

    descriptions = dataset.get_descriptions(visit_id)
    for desc in descriptions:
        if desc['desc_id'] == desc_id:
            desc_text = desc['description']
            annot_id = desc.get('annot_id')
            break
    if desc_text == '' or annot_id == '':
        print(f"error: desc_id {desc_id} not in descriptions, skip")
        exit()
    gt_mask = dataset.get_grouped_annotation(visit_id, desc_id)
    gt_mask = torch.tensor(gt_mask)

    laser_scan_path = dataset.get_data_asset_path(
        split=split,
        data_asset_identifier="laser_scan_5mm",
        visit_id=visit_id
    )
    laser_scan = o3d.io.read_point_cloud(laser_scan_path)
    cropped_laser_scan, filtered_idx_list = dataset.get_cropped_laser_scan_and_id(visit_id, laser_scan)

    cropped_indices = np.where(np.isin(filtered_idx_list, original_indices))[0]

    if len(cropped_indices) == 0:
        print(f"{mask_json_path} original_indices/cropped_indices empty, skip")
        return
    pred_mask = torch.zeros_like(gt_mask, dtype=torch.uint8)
    pred_mask[cropped_indices] = 1
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_points = torch.tensor(np.asarray(cropped_laser_scan.points), device=device, dtype=torch.float32)
    mask_points = all_points[cropped_indices]
    mean_xyz = mask_points.mean(dim=0)

    distances = torch.norm(all_points - mean_xyz, dim=1)
    within_radius_mask = (distances <= radius)
    if within_radius_mask.sum().item() < num_points:
        remaining_mask = ~within_radius_mask
        remaining_distances = distances[remaining_mask]
        remaining_indices = torch.where(remaining_mask)[0]
        num_to_add = num_points - within_radius_mask.sum().item()
        if num_to_add > 0 and len(remaining_distances) > 0:
            closest_indices = remaining_indices[torch.argsort(remaining_distances)[:num_to_add]]
            within_radius_mask[closest_indices] = True
    if within_radius_mask.sum().item() > num_points:
        selected_indices = torch.where(within_radius_mask)[0]
        selected_distances = distances[selected_indices]
        sorted_indices = torch.argsort(selected_distances)[:num_points]
        closest_indices = selected_indices[sorted_indices]
        new_within_radius_mask = torch.zeros_like(within_radius_mask, dtype=torch.bool, device=device)
        new_within_radius_mask[closest_indices] = True
        within_radius_mask = new_within_radius_mask
    filtered_points = all_points[within_radius_mask].cpu().numpy()
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    if cropped_laser_scan.has_colors():
        all_colors = np.asarray(cropped_laser_scan.colors)
        filtered_colors = all_colors[within_radius_mask.cpu().numpy()]
        filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
    os.makedirs(save_dir, exist_ok=True)
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
    print(f"Saved json to {json_path}")
    ply_path = os.path.join(save_dir, "filtered_point_cloud.ply")
    o3d.io.write_point_cloud(ply_path, filtered_point_cloud)
    print(f"Saved ply to {ply_path}")
    pred_mask_global_path = os.path.join(save_dir, "pred_mask_global.npy")
    np.save(pred_mask_global_path, pred_mask.cpu().numpy())
    print(f"Saved pred_mask_global to {pred_mask_global_path}")
    pred_mask_local_path = os.path.join(save_dir, "pred_mask_local.npy")
    np.save(pred_mask_local_path, pred_mask[within_radius_mask.cpu().numpy()])
    print(f"Saved pred_mask_local to {pred_mask_local_path}")
    gt_mask_global_path = os.path.join(save_dir, "gt_mask_global.npy")
    np.save(gt_mask_global_path, gt_mask.cpu().numpy())
    print(f"Saved gt_mask_global to {gt_mask_global_path}")
    gt_mask_local_path = os.path.join(save_dir, "gt_mask_local.npy")
    np.save(gt_mask_local_path, gt_mask[within_radius_mask.cpu().numpy()])
    print(f"Saved gt_mask_local to {gt_mask_local_path}")


def main():
    parser = argparse.ArgumentParser(description='Chunk point cloud from mask_result.json')
    parser.add_argument('--root_dir', type=str, default='path/to/data', help='Data root')
    parser.add_argument('--clip_dir', type=str, default='path/to/lift', help='Lift output root')
    parser.add_argument('--raw_data_dir', type=str, default='path/to/raw_data', help='Raw point cloud root')
    parser.add_argument('--save_dir', type=str, default='path/to/processed_sam2', help='Save root')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Split')
    parser.add_argument('--radius', type=float, default=0.15, help='Radius')
    parser.add_argument('--num_points', type=int, default=65536, help='Num points')
    args = parser.parse_args()
    splits = ['val']
    for split in splits:
        split_dir = args.clip_dir
        if not os.path.exists(split_dir):
            print(f"Dir not found: {split_dir}")
            continue

        dataset = AffordanceDataset(
            root_dir=args.root_dir,
            split=args.split,
            use_processed_data=False,
            use_division=True
        )
        for visit_id in tqdm(os.listdir(split_dir), desc=split):
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
                    save_dir = os.path.join(args.save_dir, split, visit_id, scan_id, desc_id)
                    try:
                        process_mask_and_pointcloud_from_json(
                            mask_json_path, dataset, split, save_dir, visit_id, scan_id, desc_id, radius=args.radius, num_points=args.num_points
                        )
                    except Exception as e:
                        print(f"Error processing {mask_json_path}: {str(e)}")
                        continue
    print("All done.")

if __name__ == "__main__":
    main()

