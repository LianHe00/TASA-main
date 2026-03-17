import os
import json
import torch
from PIL import Image
import torch.nn as nn
import clip
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

def process_clip(data_root, split):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32")
    print("CLIP model loaded successfully")
    model.cuda().eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    benchmark_dir = os.path.join(data_root, "benchmark_file_lists")
    split_file = os.path.join(benchmark_dir, f"{split}_set.csv")
    if not os.path.exists(split_file):
        print(f"Error: split file not found {split_file}")
        return
    print(f"Reading split file: {split_file}")
    df = pd.read_csv(split_file)
    print(f"Found {len(df)} scene(s)")
    df['visit_id'] = df['visit_id'].astype(str)
    df_single = df.drop_duplicates(subset=['visit_id'], keep='first')
    print(f"Unique visit_id: {len(df_single)}")
    for index, row in tqdm(df_single.iterrows(), total=len(df_single), desc='visit_id'):
        visit_id = row['visit_id']
        visit_rows = df[df['visit_id'] == visit_id]
        print(f"processing visit_id: {visit_id}")
        desc_path = os.path.join(data_root, split, visit_id, f"{visit_id}_descriptions.json")
        affordance_path = os.path.join('path/to/affordance_result', split, f"{visit_id}_affordance.json")
        descid2affordance = {}
        if os.path.exists(affordance_path):
            with open(affordance_path, 'r') as f:
                affordance_data = json.load(f)
            for item in affordance_data:
                descid2affordance[item['desc_id']] = item['affordance']
        else:
            print(f"Warning: affordance file not found {affordance_path}")
        if os.path.exists(desc_path):
            with open(desc_path, 'r') as f:
                desc_data = json.load(f)
            desc_list = desc_data['descriptions']
            print(f"Found {len(desc_list)} description(s)")
        else:
            print(f"Warning: description file not found {desc_path}")
            continue
        for _, row in visit_rows.iterrows():
            video_id = str(row['video_id'])
            image_path = os.path.join(data_root, split, visit_id, video_id, "hires_wide")
            print(f"processing visit_id: {visit_id} video_id: {video_id}")
            image_files = []
            if os.path.exists(image_path):
                for filename in os.listdir(image_path):
                    if filename.lower().endswith('.jpg'):
                        full_path = os.path.join(image_path, filename)
                        image_files.append(full_path)
            else:
                print(f"Warning: image dir not found {image_path}")
                continue
            all_images = image_files
            if len(all_images) == 0:
                print(f"Warning: no images in {image_path}")
                continue
            image_inputs = []
            for img_path in all_images:
                image = Image.open(img_path).convert('RGB')
                image_input = preprocess(image).unsqueeze(0)
                image_inputs.append(image_input)
            image_input = torch.cat(image_inputs, dim=0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features_cpu = image_features.cpu().numpy()
            results = []
            for desc in desc_list:
                print(f"processing desc_id: {desc['desc_id']}")
                desc_id = desc['desc_id']
                description = desc['description']
                affordance = descid2affordance.get(desc_id, None)
                if affordance and affordance.strip():
                    enhanced_text = f"{description} [AFFORDANCE] {affordance} {affordance} {affordance}"
                else:
                    enhanced_text = description
                text_tokens = clip.tokenize([enhanced_text]).to(device)
                with torch.no_grad():
                    text_features = model.encode_text(text_tokens).float()
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features_cpu = text_features.cpu().numpy()
                similarity = text_features_cpu @ image_features_cpu.T
                similarity_scores = similarity.squeeze()
                top_10_indices = np.argsort(similarity_scores)[::-1][:10]
                top10_image_names = [os.path.basename(all_images[idx]) for idx in top_10_indices]
                result_item = {
                    'visit_id': visit_id,
                    'video_id': video_id,
                    'desc_id': desc_id,
                    'description': description,
                    'affordance': affordance,
                    'image_name': top10_image_names
                }
                results.append(result_item)
            if results:
                test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clipwithaffordance_output', split, visit_id)
                os.makedirs(test_dir, exist_ok=True)
                video_result_path = os.path.join(test_dir, f"{video_id}_result.json")
                with open(video_result_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"Done, saved to {video_result_path}")

def main():
    parser = argparse.ArgumentParser(description='CLIP affordance dataset processing')
    parser.add_argument('--data_root', type=str, required=True, help='Data root path')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True, help='Dataset split')
    args = parser.parse_args()
    process_clip(args.data_root, args.split)

if __name__ == '__main__':
    main()
