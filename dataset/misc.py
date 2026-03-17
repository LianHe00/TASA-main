from typing import Dict, List
import torch
from einops import rearrange

def collate_fn_general(batch: List) -> Dict:
    """ General collate function used for dataloader.
    """
    batch_data = {key: [d[key] for d in batch] for key in batch[0]}

    variable_shape_keys = ['original_indices', 'c_original_pc_xyz', 'c_original_pc_feat']
    variable_shape_keys_2 = ['gt_mask_global', 'pred_mask_global']
    

    
    for key in batch_data:
        if torch.is_tensor(batch_data[key][0]):
            if key in variable_shape_keys or key in variable_shape_keys_2:
                continue
            else:
                batch_data[key] = torch.stack(batch_data[key])
    return batch_data