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

def process_one_molmo_mask(npz_path, data_root, split, visit_id, video_id, desc_id, all_results, output_dir=None):
    laser_scan_path = os.path.join(data_root, "raw_data", split, visit_id, f"{visit_id}_laser_scan.ply")
    traj_path = os.path.join(data_root, "raw_data", split, visit_id, video_id, "hires_poses.traj")

    if not os.path.exists(laser_scan_path):
        print(f"Point cloud file not found: {laser_scan_path}")
        return
    
    pcd = o3d.io.read_point_cloud(laser_scan_path)
    original_points = np.asarray(pcd.points)
    original_colors = np.asarray(pcd.colors)

    parser = DataParser(os.path.join(data_root, "raw_data", split))
    crop_indices = parser.get_crop_mask(visit_id, return_indices=True)

    points = original_points[crop_indices]
    colors = original_colors[crop_indices]

    npz = np.load(npz_path)
    masks = npz['masks']
    points_2d = npz['points'] if 'points' in npz else None

    print(f"Processing mask file: {npz_path}")
    print(f"Masks: {len(masks)}, 2D points: {len(points_2d) if points_2d is not None else 0}")

    basename = os.path.basename(npz_path)

    if basename.endswith('_crop_mask_data.npz'):
        if basename.startswith('frame'):
            name_part = basename.replace('frame', '').replace('_crop_mask_data.npz', '')
            parts = name_part.split('_')
            if len(parts) >= 3:
                frame_id = parts[-1]
            else:
                print(f"Cannot parse frame_id from: {basename}")
                return
        else:
            name_part = basename.replace('_crop_mask_data.npz', '')
            parts = name_part.split('_')
            if len(parts) >= 2:
                frame_id = parts[-1]
            else:
                print(f"Cannot parse frame_id from: {basename}")
                return
    elif basename.endswith('_mask_data.npz'):
        name_part = basename.replace('_mask_data.npz', '')
        parts = name_part.split('_')
        if len(parts) >= 2:
            frame_id = parts[-1]
        else:
            print(f"Cannot parse frame_id from: {basename}")
            return
    else:
        print(f"Unsupported filename format: {basename} (expected *_crop_mask_data.npz or *_mask_data.npz)")
        return

    depth_dir = os.path.join(data_root, "raw_data", split, visit_id, video_id, "hires_depth")
    depth_path = os.path.join(depth_dir, f"{video_id}_{frame_id}.png")
    if not os.path.exists(depth_path):
        print(f"Depth file not found: {depth_path}")
        return
    
    depth = parser.read_depth_frame(depth_path)
    h, w = depth.shape

    intrinsics_dir = os.path.join(data_root, "raw_data", split, visit_id, video_id, "hires_wide_intrinsics")
    intrinsics_files = [f for f in os.listdir(intrinsics_dir) if f.endswith('.pincam')]
    intrinsic_path = None
    for f in intrinsics_files:
        if frame_id in f:
            intrinsic_path = os.path.join(intrinsics_dir, f)
            break
    
    if intrinsic_path is None:
        print(f"Intrinsics not found: {intrinsics_dir} (frame_id={frame_id})")
        return

    intrinsic = parser.read_camera_intrinsics(intrinsic_path, format="matrix")

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
        print(f"No suitable camera pose for frame_id: {frame_id}")
        return

    if torch.cuda.is_available():
        proc_pcd = torch.tensor(points).cuda()
        device = "cuda"
    else:
        proc_pcd = torch.tensor(points)
        device = "cpu"
    
    mask_threshold = 0.5

    for mask_idx, mask in enumerate(masks):
        print(f"Processing mask {mask_idx}")

        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        
        whole_mask = np.ones(depth.shape)

        mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(torch.float)
        whole_mask_tensor = torch.tensor(whole_mask).unsqueeze(0).unsqueeze(0).to(torch.float)
        if torch.cuda.is_available():
            mask_tensor = mask_tensor.cuda()
            whole_mask_tensor = whole_mask_tensor.cuda()
        
        mapper = PointCloudToImageMapper((w, h))
        mapping_fo = mapper.compute_multi_masked_mapping(
            pose,
            proc_pcd,
            torch.stack([mask_tensor.squeeze(), whole_mask_tensor.squeeze()], dim=0),
            depth,
            intrinsic,
            device
        )

        valid_f = mapping_fo[0, :, -1] == 1
        valid_count = np.count_nonzero(valid_f)

        if valid_count > 0:
            valid_fy = mapping_fo[0, valid_f, 0].astype(int)
            valid_fx = mapping_fo[0, valid_f, 1].astype(int)
            valid_point_indices = np.where(valid_f)[0]

            mask_values = mask[valid_fy, valid_fx]
            mask_points = mask_values > mask_threshold

            if np.sum(mask_points) > 0:
                mask_point_indices = valid_point_indices[mask_points]

                cropped_mask_indices = sorted(mask_point_indices)
                original_mask_indices = [crop_indices[idx] for idx in cropped_mask_indices]

                mask_points_3d = original_points[original_mask_indices]
                mask_colors_3d = original_colors[original_mask_indices]
                mean_xyz = np.mean(mask_points_3d, axis=0).tolist()

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

                if points_2d is not None and len(points_2d) > mask_idx:
                    result_data["points_2d"] = points_2d[mask_idx].tolist() if hasattr(points_2d[mask_idx], 'tolist') else points_2d[mask_idx]

                all_results.append(result_data)

                if output_dir is not None:
                    temp_output_path = os.path.join(output_dir, f"temp_{visit_id}_{video_id}_{desc_id}.json")
                    with open(temp_output_path, 'w') as f:
                        json.dump(all_results, f, indent=2, ensure_ascii=False)

                print(f"Processed mask {mask_idx}, points: {len(original_mask_indices):,}")
            else:
                print(f"Mask {mask_idx} has no points above threshold {mask_threshold}")
        else:
            print(f"Mask {mask_idx} has no valid point cloud projection")

    print(f"Done: {npz_path}")


def process_desc_directory(desc_dir, data_root, split, visit_id, video_id, desc_id, all_results, output_dir=None):
    if not os.path.exists(desc_dir):
        print(f"Warning: desc dir not found: {desc_dir}")
        return

    npz_files = [f for f in os.listdir(desc_dir) if f.endswith('.npz')]

    if len(npz_files) == 0:
        print(f"Warning: no npz files in {desc_dir}")
        return

    print(f"Processing {desc_id}: {len(npz_files)} npz files")

    for npz_file in npz_files:
        npz_path = os.path.join(desc_dir, npz_file)
        process_one_molmo_mask(npz_path, data_root, split, visit_id, video_id, desc_id, all_results, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Molmo 2D-to-3D mask lifting tool')
    parser.add_argument('--data_root', type=str, required=True, help='Path to data root')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'], help='Dataset split')
    parser.add_argument('--molmo_root', type=str, default='path/to/molmo_merge', help='Path to molmo_merge output')
    parser.add_argument('--output_dir', type=str, default='path/to/lift', help='Path to save lift results')
    parser.add_argument('--visit_id', type=str, default=None, help='Specific visit_id (default: all)')
    parser.add_argument('--video_id', type=str, default=None, help='Specific video_id (default: all)')
    parser.add_argument('--desc_id', type=str, default=None, help='Specific desc_id (default: all)')
    parser.add_argument('--real_time_save', action='store_true', help='Enable real-time save')
    args = parser.parse_args()

    data_root = args.data_root
    split = args.split
    molmo_root = args.molmo_root
    output_dir = args.output_dir

    if args.visit_id:
        visit_ids = [args.visit_id]
    else:
        visit_ids = [d for d in os.listdir(molmo_root)
                    if os.path.isdir(os.path.join(molmo_root, d))]
        visit_ids = sorted(visit_ids)

    print(f"Processing {len(visit_ids)} visit_id(s): {visit_ids}")
    
    all_results = []
    real_time_save = args.real_time_save

    for visit_id in tqdm(visit_ids, desc='visit_id'):
        visit_dir = os.path.join(molmo_root, visit_id)
        if not os.path.exists(visit_dir):
            print(f"Warning: visit dir not found: {visit_dir}")
            continue

        if args.video_id:
            video_ids = [args.video_id]
        else:
            video_ids = [d for d in os.listdir(visit_dir)
                        if os.path.isdir(os.path.join(visit_dir, d))]

        print(f"Visit {visit_id}: {len(video_ids)} video(s)")

        for video_id in video_ids:
            video_dir = os.path.join(visit_dir, video_id)
            if not os.path.exists(video_dir):
                print(f"Warning: video dir not found: {video_dir}")
                continue

            if args.desc_id:
                desc_ids = [args.desc_id]
            else:
                desc_ids = [d for d in os.listdir(video_dir)
                           if os.path.isdir(os.path.join(video_dir, d))]

            for desc_id in desc_ids:
                desc_dir = os.path.join(video_dir, desc_id)
                process_desc_directory(desc_dir, data_root, split, visit_id, video_id, desc_id, all_results, output_dir if real_time_save else None)

    os.makedirs(output_dir, exist_ok=True)

    if all_results:
        if args.visit_id and args.video_id and args.desc_id:
            output_filename = f"lift_results_{args.visit_id}_{args.video_id}_{args.desc_id}.json"
        elif args.visit_id:
            output_filename = f"lift_results_{args.visit_id}.json"
        else:
            output_filename = f"lift_results_all.json"
        
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"Done. Processed {len(all_results)} mask(s). Results saved to: {output_path}")
    else:
        print("No valid mask data processed.")

    print("All done.")


if __name__ == '__main__':
    main() 