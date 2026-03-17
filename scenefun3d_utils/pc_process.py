import numpy as np
import open3d as o3d

# Combines multiple laser scans / point clouds
def combine_point_clouds(ply_filenames):
    pcd_combined = o3d.geometry.PointCloud()
    for filename in ply_filenames:

        pcd = o3d.io.read_point_cloud(filename)

        pcd_combined += pc_downsample(pcd)

    return pcd_combined

# by mht
# Downsamples a point cloud using voxel downsampling and returns mapping
def pc_downsample_with_mapping(pcd_combined, voxel_size=0.005):
    """
    Downsample point cloud and return mapping from original to downsampled points.
    Args:
        pcd_combined: original point cloud
        voxel_size: voxel size
    Returns:
        downpcd: downsampled point cloud
        point_mapping: array mapping each original point to downsampled point index
    """
    points = np.asarray(pcd_combined.points)

    voxel_indices = np.floor(points / voxel_size).astype(int)

    voxel_dict = {}
    for idx, voxel_idx in enumerate(voxel_indices):
        voxel_key = tuple(voxel_idx)
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = []
        voxel_dict[voxel_key].append(idx)

    downpcd = pcd_combined.voxel_down_sample(voxel_size=voxel_size)

    point_mapping = np.zeros(len(points), dtype=int)
    down_points = np.asarray(downpcd.points)

    for voxel_key, point_indices in voxel_dict.items():
        voxel_center = (np.array(voxel_key) + 0.5) * voxel_size
        distances = np.linalg.norm(down_points - voxel_center, axis=1)
        nearest_down_idx = np.argmin(distances)
        point_mapping[point_indices] = nearest_down_idx
    return downpcd, point_mapping

# Downsamples a point cloud using voxel downsampling
def pc_downsample(pcd_combined, voxel_size=0.005):
    downpcd = pcd_combined.voxel_down_sample(voxel_size=voxel_size)

    return downpcd

# estimate normals of a point cloud
def pc_estimate_normals(pcd, radius = 0.1, max_nn = 16):
    pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = radius, max_nn = max_nn))

    return pcd
# def crop_extraneous_points_from_point_cloud(pcd, 
#                                             dbscan_eps = 0.02, 
#                                             dbscan_min_points = 10, 
#                                             return_bbox = False,
#                                             print_debug = False):
    
#     labels = np.array(pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=print_debug))

#     max_label = labels.max()

#     if print_debug:
#         print(f"Point cloud has {max_label + 1} clusters")

#     unique_labels, label_counts = np.unique(labels, return_counts=True)
#     label_counts[unique_labels < 0] = 0

#     largest_cluster_label = unique_labels[np.argmax(label_counts)]
#     largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

#     largest_cluster_points = pcd.select_by_index(largest_cluster_indices)
    
#     # Calculate the bounding box of the largest cluster
#     bbox = largest_cluster_points.get_oriented_bounding_box()
    
#     if print_debug:
#         print(f"Initial point cloud: {pcd}")

#     pcd_cropped = pcd.crop(bbox)

#     if print_debug:
#         print(f"Point cloud after cropping: {pcd_cropped}")

#     if return_bbox:
#         bbox.color = (1, 0, 0) # change bbox color for better visualization
#         return pcd_cropped, bbox
#     else:
#         return pcd_cropped

def crop_extraneous_points_from_point_cloud(pcd, 
                                            dbscan_eps = 0.02, 
                                            dbscan_min_points = 10, 
                                            return_bbox = False,
                                            print_debug = False):
    
    labels = np.array(pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=print_debug))

    max_label = labels.max()

    if print_debug:
        print(f"Point cloud has {max_label + 1} clusters")

    unique_labels, label_counts = np.unique(labels, return_counts=True)

    label_counts[unique_labels < 0] = 0

    largest_cluster_label = unique_labels[np.argmax(label_counts)]
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

    # needs Open3D version > 0.9.0 
    # largest_cluster_points = pcd.select_by_index(largest_cluster_indices)
    
    # For Open3D version = 0.9.0
    pcd_points = np.array(pcd.points)
    pcd_points = pcd_points[largest_cluster_indices]
    largest_cluster_points = o3d.geometry.PointCloud()
    largest_cluster_points.points = o3d.utility.Vector3dVector(pcd_points)
    
    # Calculate the bounding box of the largest cluster
    bbox = largest_cluster_points.get_oriented_bounding_box()
    
    if print_debug:
        print(f"Initial point cloud: {pcd}")

    # needs Open3D version > 0.9.0 
    # pcd_cropped = pcd.crop(bbox)

    # For Open3D version = 0.9.0
    pcd_cropped = crop_point_cloud(pcd, bbox)

    if print_debug:
        print(f"Point cloud after cropping: {pcd_cropped}")

    if return_bbox:
        bbox.color = (1, 0, 0) # change bbox color for better visualization
        return pcd_cropped, bbox
    else:
        return pcd_cropped


# Crop function is malfunctioning in Open3d==0.9.0 - https://github.com/isl-org/Open3D/issues/3284
def crop_point_cloud(pcd, bbox):
    point_cloud_np = np.asarray(pcd.points)
    point_cloud_np_colors = np.asarray(pcd.colors)

    # mask = bbox.get_point_indices_within_bounding_box(pcd.points) # get_point_indices_within_bounding_box() does not work well for Open3d==0.9.0

    # Define a boolean mask to filter points within the bounding box
    mask = np.logical_and(np.all(point_cloud_np >= bbox.get_min_bound(), axis=1),
                        np.all(point_cloud_np <= bbox.get_max_bound(), axis=1))

    # Apply the mask to extract the cropped point cloud
    cropped_point_cloud = o3d.geometry.PointCloud()
    cropped_point_cloud.points = o3d.utility.Vector3dVector(point_cloud_np[mask])
    cropped_point_cloud.colors = o3d.utility.Vector3dVector(point_cloud_np_colors[mask])

    return cropped_point_cloud

# by mht
def map_labels_to_original_points(original_pcd, downsampled_pcd, downsampled_labels, method='knn', k=3):
    """
    Map labels from downsampled point cloud back to original point cloud.
    Args:
        original_pcd: original point cloud (open3d.geometry.PointCloud)
        downsampled_pcd: downsampled point cloud (open3d.geometry.PointCloud)
        downsampled_labels: labels on downsampled points (np.array)
        method: 'knn' (k-nearest neighbors) or 'voxel' (voxel mapping)
        k: number of neighbors for knn
    Returns:
        original_labels: labels mapped to original points (np.array)
    """
    original_points = np.asarray(original_pcd.points)
    downsampled_points = np.asarray(downsampled_pcd.points)

    if method == 'knn':
        pcd_tree = o3d.geometry.KDTreeFlann(downsampled_pcd)
        original_labels = np.zeros(len(original_points))

        for i, point in enumerate(original_points):
            _, idx, dist = pcd_tree.search_knn_vector_3d(point, k)
            weights = 1.0 / (np.array(dist) + 1e-10)
            weights = weights / np.sum(weights)
            original_labels[i] = np.sum(downsampled_labels[idx] * weights)

    elif method == 'voxel':
        bbox = downsampled_pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        points_count = len(downsampled_points)
        voxel_size = np.mean(extent) / np.cbrt(points_count)

        voxel_indices_original = np.floor(original_points / voxel_size).astype(int)
        voxel_indices_down = np.floor(downsampled_points / voxel_size).astype(int)

        voxel_label_map = {}
        for idx, voxel_idx in enumerate(voxel_indices_down):
            voxel_key = tuple(voxel_idx)
            voxel_label_map[voxel_key] = downsampled_labels[idx]

        original_labels = np.zeros(len(original_points))
        for i, voxel_idx in enumerate(voxel_indices_original):
            voxel_key = tuple(voxel_idx)
            if voxel_key in voxel_label_map:
                original_labels[i] = voxel_label_map[voxel_key]
            else:
                dists = np.linalg.norm(downsampled_points - original_points[i], axis=1)
                nearest_idx = np.argmin(dists)
                original_labels[i] = downsampled_labels[nearest_idx]
    
    else:
        raise ValueError(f"Unsupported method: {method}")
        
    return original_labels



def uniform_sample_fixed(points, labels=None, num_samples=1024, method='fps'):
    """
    Sample point cloud to fixed number of points, optionally with labels.
    Args:
        points: Open3D PointCloud or (N, C) NumPy array
        labels: (N,) optional labels
        num_samples: target number of points
        method: 'random' or 'fps'
    Returns:
        sampled_points: Open3D PointCloud or NumPy array (same type as input)
        sampled_labels: (num_samples,) or None
    """
    is_o3d = isinstance(points, o3d.geometry.PointCloud)
    if is_o3d:
        points_np = np.asarray(points.points)
        if points.has_colors():
            colors_np = np.asarray(points.colors)
            points_np = np.concatenate([points_np, colors_np], axis=1)
    else:
        points_np = points
    
    N = points_np.shape[0]
    
    if N >= num_samples:
        if method == 'random':
            if labels is not None and np.any(labels > 0):
                pos_indices = np.where(labels > 0)[0]
                neg_indices = np.where(labels == 0)[0]

                pos_ratio = len(pos_indices) / N
                n_pos_samples = int(num_samples * pos_ratio)
                n_pos_samples = min(n_pos_samples, len(pos_indices))
                n_neg_samples = num_samples - n_pos_samples

                sampled_pos = np.random.choice(pos_indices, n_pos_samples, replace=False)
                sampled_neg = np.random.choice(neg_indices, n_neg_samples, replace=False)
                idx = np.concatenate([sampled_pos, sampled_neg])
                np.random.shuffle(idx)
            else:
                idx = np.random.choice(N, num_samples, replace=False)
        elif method == 'fps':
            xyz = points_np[:, :3]
            idx = farthest_point_sampling_numpy(xyz, num_samples)
        else:
            raise ValueError("Unsupported method")
            
    else:
        if labels is not None and np.any(labels > 0):
            pos_indices = np.where(labels > 0)[0]
            neg_indices = np.where(labels == 0)[0]

            extra_samples = num_samples - N
            pos_ratio = len(pos_indices) / N
            n_pos_extra = int(extra_samples * pos_ratio)
            n_neg_extra = extra_samples - n_pos_extra

            pos_extra_idx = np.random.choice(pos_indices, n_pos_extra, replace=True)
            neg_extra_idx = np.random.choice(neg_indices, n_neg_extra, replace=True)
            idx = np.arange(N)
            extra_idx = np.concatenate([pos_extra_idx, neg_extra_idx])
            idx = np.concatenate([idx, extra_idx])
        else:
            extra_idx = np.random.choice(N, num_samples - N, replace=True)
            idx = np.concatenate([np.arange(N), extra_idx])

    sampled_points_np = points_np[idx]

    if is_o3d:
        sampled_pc = o3d.geometry.PointCloud()
        sampled_pc.points = o3d.utility.Vector3dVector(sampled_points_np[:, :3])
        if points.has_colors():
            sampled_pc.colors = o3d.utility.Vector3dVector(sampled_points_np[:, 3:])
        sampled_points = sampled_pc
    else:
        sampled_points = sampled_points_np

    sampled_labels = labels[idx] if labels is not None else None

    return sampled_points, sampled_labels


def farthest_point_sampling_numpy(points, n_samples):
    """
    NumPy FPS (Farthest Point Sampling).
    Args:
        points: (N, 3) point cloud
        n_samples: number of points to sample
    Returns:
        indices: (n_samples,) sampled point indices
    """
    N = points.shape[0]
    if n_samples > N:
        print(f"Warning: n_samples ({n_samples}) > N ({N}), setting n_samples = N")
        n_samples = N

    centroids = np.zeros(n_samples, dtype=np.int32)
    distance = np.full(N, np.inf, dtype=np.float32)
    farthest = np.random.randint(0, N)

    for i in range(n_samples):
        centroids[i] = farthest
        centroid = points[farthest]

        diff = points - centroid
        dist = np.sum(diff * diff, axis=1).astype(np.float32)

        mask = dist < distance
        distance[mask] = dist[mask]

        if i < n_samples - 1:
            farthest = np.argmax(distance)
    
    return centroids

def voxel_sample_points(points, labels=None, num_samples=1024):
    """
    Random sampling for point cloud (for large-scale data).
    Args:
        points: Open3D PointCloud or (N, C) NumPy array
        labels: (N,) optional labels
        num_samples: target number of points
    Returns:
        sampled_points: sampled point cloud (same format as input)
        sampled_labels: (num_samples,) or None
        point_mapping: mapping from original to sampled points
    """
    is_o3d = isinstance(points, o3d.geometry.PointCloud)
    if is_o3d:
        points_np = np.asarray(points.points)
        has_colors = points.has_colors()
        if has_colors:
            colors_np = np.asarray(points.colors)
            points_np = np.concatenate([points_np, colors_np], axis=1)
    else:
        points_np = points
        has_colors = points.shape[1] > 3

    N = len(points_np)

    if N > num_samples:
        indices = np.random.choice(N, num_samples, replace=False)
        sampled_points = points_np[indices]
    else:
        indices = np.random.choice(N, num_samples, replace=True)
        sampled_points = points_np[indices]

    point_mapping = np.zeros(N, dtype=np.int32)

    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(sampled_points[:, :3])
    distances, indices = nbrs.kneighbors(points_np[:, :3])
    point_mapping = indices.ravel()

    if labels is not None:
        sampled_labels = np.zeros(num_samples)
        for i in range(num_samples):
            mask = point_mapping == i
            if np.any(mask):
                unique_labels, counts = np.unique(labels[mask], return_counts=True)
                sampled_labels[i] = unique_labels[np.argmax(counts)]
    else:
        sampled_labels = None

    if is_o3d:
        output_pcd = o3d.geometry.PointCloud()
        output_pcd.points = o3d.utility.Vector3dVector(sampled_points[:, :3])
        if has_colors:
            output_pcd.colors = o3d.utility.Vector3dVector(sampled_points[:, 3:])
        sampled_points = output_pcd

    return sampled_points, sampled_labels, point_mapping

def bucket_fps_kdline_sampling(points, num_samples=1024):
    """
    Bucket sampling + FPS for large-scale point clouds.
    Args:
        points: NumPy array
        num_samples: target number of points
    Returns:
        kdline_fps_samples_idx: sampled point indices
    """
    is_o3d = isinstance(points, o3d.geometry.PointCloud)
    if is_o3d:
        points_np = np.asarray(points.points)
    else:
        points_np = points

    import fpsample
    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(points_np, num_samples, h=3)

    return kdline_fps_samples_idx
