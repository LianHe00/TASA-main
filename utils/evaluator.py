import os
import sys

sys.path.append(os.getcwd())
import json
import pickle
from typing import List, Optional

import numpy as np
import torch
import open3d as o3d

from utils.metrics import (
    compute_3d_ap,
    compute_3d_ar,
    compute_mean_iou,
    compute_mean_recalls,
    compute_scores,
)

from .metrics import *
def viz_3d_masks(pcd: np.array, gt_mask: np.array, pred_mask: np.array, viz_save_path) -> None:
    """
    Visualizes predicted masks agains a ground truth. Keys:
    Masks are represented as binary masks
    - gt mask only is visualized in blue
    - pred mask only is visualized in red
    - overlap part is visualized in green
    """
    assert (pcd.shape[0] == gt_mask.shape[0]) and (
        pred_mask.shape[0] == gt_mask.shape[0]
    ), " Mask size do not correspond."


    # TODO: assert that colors are in range 0..1 as required by open3d

    xyz, rgb = pcd[:, :3].copy(), pcd[:, 3:].copy()

    # compute overlap masks
    overlap = np.where(np.logical_and(gt_mask, pred_mask), 1.0, 0.0)

    gt_idx = np.nonzero(gt_mask)[0]
    pred_idx = np.nonzero(pred_mask)[0]
    overlap_idx = np.nonzero(overlap)[0]

    # ground truth only in blue
    rgb[gt_idx] = np.asarray([0.0, 0.0, 1.0])
    # pred only in red
    rgb[pred_idx] = np.asarray([1.0, 0.0, 0.0])
    # # # commmon (IoU) in green
    if overlap_idx.shape[0] != 0:
        rgb[overlap_idx] = np.asarray([0.0, 1.0, 0.0])

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d_pcd.colors = o3d.utility.Vector3dVector(rgb)

    o3d.io.write_point_cloud(viz_save_path, o3d_pcd)
    print(f"Point cloud saved as {viz_save_path}")

    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=1.0,  # The size of the axes, adjust based on your scene
    #     origin=[0, 0, 0],  # The origin of the coordinate frame
    # )

    # o3d.visualization.draw_geometries([o3d_pcd])

class Segment3DEvaluator(object):
    """
    Helper class used to evaluate mask metrics
    """

    def __init__(self, exp_tag: str, viz_dir: str, threshold: float = 0.5):

        super().__init__()
        self.exp_tag = exp_tag
        self.threshold = threshold
        self.AP_TH = torch.linspace(0.5, 0.95, 10)
        self.metrics = {}
        self.viz_dir = viz_dir

        self.metrics["visit_id"] = []
        self.metrics["annot_id"] = []
        self.metrics["pred_count"] = []
        self.metrics["gt_count"] = []
        self.metrics["Prc"] = []
        self.metrics["mAP"] = []
        self.metrics["AP25"] = []
        self.metrics["AP50"] = []
        self.metrics["Rec"] = []
        self.metrics["mAR"] = []
        self.metrics["AR25"] = []
        self.metrics["AR50"] = []
        self.metrics["mIoU"] = []
        

    def register(
        self,
        visit_ids: List[str],
        annot_ids: List[str],
        gt_masks: Tensor,
        pred_masks: Tensor,
        full_pcd: np.ndarray,
        device: torch.device,
    ):
        """
        Register metrics mAP,AP50,AP25,mAR,AR50,AR25
        """

        assert (
            gt_masks.shape == pred_masks.shape
        ), f"Shapes not correponding: {gt_masks.shape} ad {pred_masks.shape}"

        pred_masks = (pred_masks > self.threshold).float()


        # beacuse function only processes lists
        if len(gt_masks.shape) == 1:
            gt_masks = gt_masks.unsqueeze(0)
            pred_masks = pred_masks.unsqueeze(0)

        # compute ap and relative recalls
        valid_pred = [torch.count_nonzero(pred_mask).item() for pred_mask in pred_masks]
        valid_gt = [torch.count_nonzero(gt_mask).item() for gt_mask in gt_masks]
        ap_i = compute_3d_ap(gt_masks, pred_masks)
        ious = compute_mean_iou(gt_masks, pred_masks)
        map = compute_mean_recalls(ap_i.to(device), self.AP_TH.to(device))
        ap_rec = compute_scores(ap_i, [0.25, 0.50])
        ap_50 = ap_rec[0.50]
        ap_25 = ap_rec[0.25]

        # compute ar and relative recalls
        ar_i = compute_3d_ar(gt_masks, pred_masks)
        mar = compute_mean_recalls(ar_i.to(device), self.AP_TH.to(device))
        ar_rec = compute_scores(ar_i, [0.25, 0.50])
        ar_50 = ar_rec[0.50]
        ar_25 = ar_rec[0.25]
        self.metrics["visit_id"].extend(visit_ids)
        self.metrics["annot_id"].extend(annot_ids)
        self.metrics["pred_count"].extend(valid_pred)
        self.metrics["gt_count"].extend(valid_gt)

        self.metrics["Prc"].extend(list(ap_i.cpu().numpy().astype(np.float64)))
        self.metrics["mAP"].extend(list(map.cpu().numpy().astype(np.float64)))
        self.metrics["AP50"].extend(list(ap_50.cpu().numpy().astype(np.float64)))
        self.metrics["AP25"].extend(list(ap_25.cpu().numpy().astype(np.float64)))

        self.metrics["Rec"].extend(list(ar_i.cpu().numpy().astype(np.float64)))
        self.metrics["mAR"].extend(list(mar.cpu().numpy().astype(np.float64)))
        self.metrics["AR50"].extend(list(ar_50.cpu().numpy().astype(np.float64)))
        self.metrics["AR25"].extend(list(ar_25.cpu().numpy().astype(np.float64)))

        self.metrics["mIoU"].extend(list(ious.cpu().numpy().astype(np.float64)))


        # viz_save_path = os.path.join(
        #     self.viz_dir,
        #     f"{visit_ids[0]}_{annot_ids[0]}.ply",
        # )

        # viz_3d_masks(
        #     full_pcd,
        #     gt_masks[0].numpy(),
        #     pred_masks[0].numpy(),
        #     viz_save_path
        # )

    def save(self, file):
        to_save = dict()
        for k, v in self.metrics.items():
            to_save[k] = v  # assumes is a list already
        json.dump(to_save, file)

    def get_means(self):
        """
        Returns mean of each metric registered so far
        """
        means = {}
        for name, value in self.metrics.items():
            if name not in ["visit_id", "annot_id"]:
                mean = np.asarray(value).mean()
                means[name] = mean

        return means

    def get_latex_str(self):
        """
        Returns mean of each metric in format for a latex table
        """

        means = self.get_means()

        latex_str = f"{self.exp_tag} & {means['mAP']*100:.2f} & {means['AP50']*100:.2f} & {means['AP25']*100:.2f} & {means['mAR']*100:.2f} & {means['AR50']*100:.2f} & {means['AR25']*100:.2f} & {means['mIoU']*100:.2f} \\\\ \n"

        return latex_str
