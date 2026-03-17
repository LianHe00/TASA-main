import os
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

# python test_molmo_merge.py

# 输入输出路径
MOLMO_ROOT = 'molmo_output'
CROPINFO_ROOT = 'seg_image/point_clipwithaffordance_output/val'
BIGIMG_ROOT = '/home/helian/code/affseg/data/raw_data/val'
MERGE_ROOT = 'molmo_merge'


def merge_mask_to_bigimg(molmo_mask, crop_info, bigimg_shape):
    """
    将小图mask映射回大图。
    molmo_mask: numpy数组，shape=(h, w)
    crop_info: dict，包含left, upper, width, height
    bigimg_shape: (H, W)
    返回: 大图尺寸的mask
    """
    mask_big = np.zeros(bigimg_shape, dtype=molmo_mask.dtype)
    l, u, w, h = crop_info['left'], crop_info['upper'], crop_info['width'], crop_info['height']
    # 若mask尺寸与crop尺寸不一致，需resize
    if molmo_mask.shape != (h, w):
        molmo_mask = np.array(Image.fromarray(molmo_mask).resize((w, h), resample=Image.NEAREST))
    mask_big[u:u+h, l:l+w] = molmo_mask
    return mask_big


def process_one_desc(visit_id, video_id, desc_id, files):
    # 1. 找到所有npz分割结果
    npz_files = [f for f in files if f.endswith('_mask_data.npz')]
    if not npz_files:
        return
    
    print(f"处理 {visit_id}/{video_id}/{desc_id}: 找到 {len(npz_files)} 个npz文件")
    
    # 2. 找到所有裁剪json
    cropinfo_dir = os.path.join(CROPINFO_ROOT, visit_id, video_id, desc_id)
    crop_jsons = [f for f in os.listdir(cropinfo_dir) if f.endswith('_crop.json')]
    if not crop_jsons:
        return
    
    print(f"  找到 {len(crop_jsons)} 个裁剪json文件")
    
    crop_json_map = {}
    # 建立小图名到json的映射（去掉扩展名和_crop后缀）
    for cj in crop_jsons:
        base = cj.replace('_crop.json', '')
        crop_json_map[base] = cj
    
    total_masks_processed = 0
    total_images_generated = 0
    
    # 3. 处理每个npz
    for npz_file in npz_files:
        # 尝试从npz文件名中提取小图名
        # 例如frame3_42445633_58101.895_crop_mask_data.npz -> frame3_42445633_58101.895
        # 或42445633_58114.790_crop_mask_data.npz -> 42445633_58114.790
        base_name = npz_file.replace('_mask_data.npz', '')
        if base_name.endswith('_crop'):
            base_name = base_name[:-5]
        # 优先精确匹配
        crop_json = crop_json_map.get(base_name)
        if not crop_json:
            # 尝试模糊匹配（只要_crop.json文件名在npz文件名里即可）
            for k, v in crop_json_map.items():
                if k in base_name:
                    crop_json = v
                    break
        if not crop_json:
            print(f"未找到对应裁剪json: {npz_file}")
            continue
        with open(os.path.join(cropinfo_dir, crop_json), 'r') as f:
            crop_info = json.load(f)
        # 读取大图
        bigimg_path = crop_info['original_image']
        bigimg = Image.open(bigimg_path)
        bigimg_shape = (bigimg.height, bigimg.width)
        # 读取mask
        npz_path = os.path.join(MOLMO_ROOT, visit_id, video_id, desc_id, npz_file)
        data = np.load(npz_path)
        masks = data['masks']
        
        print(f"    处理 {npz_file}: 包含 {masks.shape[0]} 个mask")
        
        # 合成mask
        merged_masks = []
        for i in range(masks.shape[0]):
            merged_mask = merge_mask_to_bigimg(masks[i], crop_info, bigimg_shape)
            merged_masks.append(merged_mask)
        merged_masks = np.stack(merged_masks, axis=0)
        
        # 保存
        save_dir = os.path.join(MERGE_ROOT, visit_id, video_id, desc_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # 为每个mask生成对应的可视化图片
        bigimg_name = os.path.splitext(os.path.basename(crop_info['original_image']))[0]
        bigimg = Image.open(bigimg_path).convert('RGB')
        
        for i, mask in enumerate(merged_masks):
            # 生成带mask编号的可视化图片
            bigimg_jpg_path = os.path.join(save_dir, f"{bigimg_name}_mask_{i:03d}.jpg")
            mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255).convert('L')
            bigimg_np = np.array(bigimg)
            mask_np = np.array(mask_img)
            bigimg_np[mask_np > 0] = [255, 0, 0]  # 红色显示mask区域
            Image.fromarray(bigimg_np).save(bigimg_jpg_path)
        

        
        # 仍然保存npz
        save_npz = os.path.join(save_dir, f"{bigimg_name}_mask_data.npz")
        np.savez_compressed(save_npz, masks=merged_masks)
        
        total_masks_processed += masks.shape[0]
        total_images_generated += masks.shape[0]
    
    print(f"  完成处理: 总共处理了 {total_masks_processed} 个mask，生成了 {total_images_generated} 张可视化图片")


def main():
    for visit_id in tqdm(os.listdir(MOLMO_ROOT), desc='visit_id'):
        visit_path = os.path.join(MOLMO_ROOT, visit_id)
        if not os.path.isdir(visit_path):
            continue
        for video_id in os.listdir(visit_path):
            video_path = os.path.join(visit_path, video_id)
            if not os.path.isdir(video_path):
                continue
            for desc_id in os.listdir(video_path):
                desc_path = os.path.join(video_path, desc_id)
                if not os.path.isdir(desc_path):
                    continue
                files = os.listdir(desc_path)
                process_one_desc(visit_id, video_id, desc_id, files)

if __name__ == '__main__':
    main()
