import os
import argparse
import json
from PIL import Image, ImageDraw
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

# CUDA_VISIBLE_DEVICES=6 python qwen_seg_image.py --data_root qwen2/point_clipwithaffordance_output --split val --size 540 540

# 用GPU裁剪图片并保存

def crop_image_gpu(image_path, center_x, center_y, width, height, output_dir, image_name_prefix=None, device='cuda'):
    image = Image.open(image_path).convert('RGB')
    img_w, img_h = image.size
    img_tensor = TF.to_tensor(image).to(device)  # [C, H, W]
    left = int(center_x - width / 2)
    upper = int(center_y - height / 2)
    right = int(center_x + width / 2)
    lower = int(center_y + height / 2)
    left = max(0, left)
    upper = max(0, upper)
    right = min(img_w, right)
    lower = min(img_h, lower)
    cropped = img_tensor[:, upper:lower, left:right]
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    if image_name_prefix:
        name = image_name_prefix + '_' + name
    output_path = os.path.join(output_dir, f"{name}_crop{ext}")
    cropped_pil = TF.to_pil_image(cropped.cpu())
    cropped_pil.save(output_path)
    # 保存映射信息
    meta = {
        "left": left,
        "upper": upper,
        "width": right - left,
        "height": lower - upper,
        "original_image": image_path
    }
    meta_path = os.path.splitext(output_path)[0] + '.json'
    with open(meta_path, 'w') as f:
        import json
        json.dump(meta, f)
    # 标记原图（在CPU上用Pillow画）
    marked_image = image.copy()
    draw = ImageDraw.Draw(marked_image)
    r = max(2, min(width, height) // 20)
    draw.ellipse((center_x - r, center_y - r, center_x + r, center_y + r), fill='red', outline='red')
    draw.rectangle((left, upper, right, lower), outline='green', width=3)
    marked_output_path = os.path.join(output_dir, f"{name}_marked{ext}")
    marked_image.save(marked_output_path)

def process_all_images(data_root, split, width, height, output_root, device):
    split_dir = os.path.join(data_root, split)
    # 先统计总任务数
    json_files = []
    for visit_id in os.listdir(split_dir):
        visit_path = os.path.join(split_dir, visit_id)
        if not os.path.isdir(visit_path):
            continue
        for fname in os.listdir(visit_path):
            if fname.endswith('_point.json'):
                json_files.append((visit_id, fname))
    # 统计所有要处理的frame数
    total_frames = 0
    for visit_id, fname in json_files:
        video_id = fname.replace('_point.json', '')
        json_path = os.path.join(split_dir, visit_id, fname)
        with open(json_path, 'r') as f:
            data = json.load(f)
        for desc_data in data:
            frame_results = desc_data.get('frame_results', [])
            for frame in frame_results:
                if frame.get('object_found', False) and frame.get('coordinates'):
                    total_frames += 1
    # 正式处理，带进度条
    pbar = tqdm(total=total_frames, desc='处理图片')
    for visit_id, fname in json_files:
        video_id = fname.replace('_point.json', '')
        json_path = os.path.join(split_dir, visit_id, fname)
        with open(json_path, 'r') as f:
            data = json.load(f)
        for desc_data in data:
            desc_id = desc_data['desc_id']
            frame_results = desc_data.get('frame_results', [])
            for idx, frame in enumerate(frame_results):
                if not frame.get('object_found', False):
                    continue
                coordinates = frame['coordinates']
                if not coordinates:
                    continue
                image_name = frame['image_name']
                center_x, center_y = coordinates['x'], coordinates['y']
                raw_img_path = f"/home/helian/code/affseg/data/raw_data/{split}/{visit_id}/{video_id}/hires_wide/{image_name}"
                out_dir = os.path.join(output_root, split, visit_id, video_id, desc_id)
                image_prefix = f"frame{idx}"
                try:
                    crop_image_gpu(raw_img_path, center_x, center_y, width, height, out_dir, image_name_prefix=image_prefix, device=device)
                except Exception as e:
                    print(f"处理图片失败: {raw_img_path}, 错误: {e}")
                pbar.update(1)
    pbar.close()

def main():
    parser = argparse.ArgumentParser(description="批量以指定点为中心裁剪图片区域（支持GPU加速）")
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录，如qwen2/point_clipwithaffordance_output')
    parser.add_argument('--split', type=str, required=True, help='数据集划分，如train/val/test')
    parser.add_argument('--size', type=int, nargs=2, required=True, metavar=('W', 'H'), help='裁剪区域宽高')
    args = parser.parse_args()
    data_root = args.data_root
    split = args.split
    width, height = args.size
    output_root = os.path.join('seg_image', os.path.basename(data_root))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    process_all_images(data_root, split, width, height, output_root, device)

if __name__ == '__main__':
    main()
