import numpy as np
import torch
import os
import json
import argparse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import glob
from collections import defaultdict
import time
from tqdm import tqdm

# CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com python test_clip4_affordance.py --data_root data/raw_data --split val

def load_and_preprocess_images(image_folder_path):
    """
    加载并预处理指定文件夹下的所有图片
    
    Args:
        image_folder_path (str): 图片文件夹路径
        
    Returns:
        tuple: (图片列表, 图片名称列表)
    """
    # 获取所有jpg图片
    image_files = glob.glob(os.path.join(image_folder_path, "*.jpg"))
    image_files.sort()  # 按文件名排序
    
    if not image_files:
        print(f"警告: 在路径 {image_folder_path} 中没有找到jpg图片")
        return [], []
    
    print(f"找到 {len(image_files)} 张图片")
    
    images = []
    image_names = []
    
    for img_path in image_files:
        try:
            # 加载图片
            image = Image.open(img_path).convert('RGB')
            images.append(image)
            
            # 获取图片名称
            image_name = os.path.basename(img_path)
            image_names.append(image_name)
            
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")
            continue
    
    return images, image_names

def get_text_embeddings(text, processor, model, device):
    """
    获取文本的嵌入向量
    
    Args:
        text (str): 输入文本
        processor: CLIP处理器
        model: CLIP模型
        device: 计算设备
        
    Returns:
        numpy.ndarray: 文本嵌入向量
    """
    # 使用处理器处理文本
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 获取文本嵌入
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    # 归一化嵌入向量
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().detach().numpy()

def get_image_embeddings(images, processor, model, device):
    """
    获取图片的嵌入向量
    
    Args:
        images (list): 图片列表
        processor: CLIP处理器
        model: CLIP模型
        device: 计算设备
        
    Returns:
        numpy.ndarray: 图片嵌入向量
    """
    if not images:
        return np.array([])
    
    # 使用处理器处理图片
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 获取图片嵌入
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # 归一化嵌入向量
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().detach().numpy()

def find_most_similar_images(text_embeddings, image_embeddings, image_names, top_k=10):
    """
    找到与文本最相似的图片
    
    Args:
        text_embeddings (numpy.ndarray): 文本嵌入向量
        image_embeddings (numpy.ndarray): 图片嵌入向量
        image_names (list): 图片名称列表
        top_k (int): 返回前k个最相似的图片
        
    Returns:
        list: 最相似图片的信息列表
    """
    if len(image_embeddings) == 0:
        return []
    
    # 计算余弦相似度
    similarities = np.dot(image_embeddings, text_embeddings.T).flatten()
    
    # 获取相似度最高的索引
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for i, idx in enumerate(top_indices):
        results.append({
            'rank': i + 1,
            'image_name': image_names[idx],
            'similarity': float(similarities[idx])  # 转换为Python float以便JSON序列化
        })
    
    return results

def load_dataset_info(data_root, split):
    """
    加载数据集信息
    
    Args:
        data_root (str): 数据根目录
        split (str): 数据集分割 (train/val/test)
        
    Returns:
        list: 包含visit_id和video_id的列表
    """
    csv_file = os.path.join(data_root, "benchmark_file_lists", f"{split}_set.csv")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"找不到文件: {csv_file}")
    
    dataset_info = []
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过标题行
            visit_id, video_id = line.strip().split(',')
            dataset_info.append({'visit_id': visit_id, 'video_id': video_id})
    
    return dataset_info

def load_descriptions(data_root, split, visit_id):
    """
    加载指定visit_id的描述信息
    
    Args:
        data_root (str): 数据根目录
        split (str): 数据集分割
        visit_id (str): 访问ID
        
    Returns:
        dict: 描述信息
    """
    desc_file = os.path.join(data_root, split, visit_id, f"{visit_id}_descriptions.json")
    
    if not os.path.exists(desc_file):
        print(f"警告: 找不到描述文件 {desc_file}")
        return None
    
    with open(desc_file, 'r') as f:
        return json.load(f)

def load_affordance_data(split, visit_id):
    """
    加载指定visit_id的affordance信息
    
    Args:
        split (str): 数据集分割
        visit_id (str): 访问ID
        
    Returns:
        dict: desc_id到affordance的映射
    """
    affordance_file = os.path.join('qwen', 'affordance_result', split, f"{visit_id}_affordance.json")
    
    if not os.path.exists(affordance_file):
        print(f"警告: 找不到affordance文件 {affordance_file}")
        return {}
    
    with open(affordance_file, 'r') as f:
        affordance_data = json.load(f)
    
    # 创建desc_id到affordance的映射
    descid2affordance = {}
    for item in affordance_data:
        descid2affordance[item['desc_id']] = item['affordance']
    
    return descid2affordance

def process_visit_id(visit_id, video_ids, descriptions, data_root, split, processor, model, device, output_dir):
    """
    处理单个visit_id的所有video_id和描述
    
    Args:
        visit_id (str): 访问ID
        video_ids (list): 该visit_id对应的video_id列表
        descriptions (dict): 描述信息
        data_root (str): 数据根目录
        split (str): 数据集分割
        processor: CLIP处理器
        model: CLIP模型
        device: 计算设备
        output_dir (str): 输出目录
        
    Returns:
        dict: 处理结果
    """
    print(f"\n处理 visit_id: {visit_id}")
    print(f"包含 {len(video_ids)} 个视频")
    
    # 创建visit_id的输出目录
    visit_output_dir = os.path.join(output_dir, split, visit_id)
    os.makedirs(visit_output_dir, exist_ok=True)
    
    if not descriptions:
        print(f"跳过 {visit_id}: 没有描述信息")
        return
    
    # 加载affordance数据
    descid2affordance = load_affordance_data(split, visit_id)
    print(f"加载了 {len(descid2affordance)} 个affordance信息")
    
    # 处理每个video_id
    for video_id in video_ids:
        print(f"  处理 video_id: {video_id}")
        
        # 构建图片路径
        image_folder_path = os.path.join(data_root, split, visit_id, video_id, "hires_wide")
        
        if not os.path.exists(image_folder_path):
            print(f"    警告: 图片文件夹不存在 {image_folder_path}")
            continue
        
        # 加载图片
        images, image_names = load_and_preprocess_images(image_folder_path)
        
        if not images:
            print(f"    警告: 没有找到图片")
            continue
        
        # 获取图片嵌入
        print(f"    获取图片嵌入...")
        image_embeddings = get_image_embeddings(images, processor, model, device)
        
        # 准备结果数据
        video_results = []
        
        # 处理每个描述
        for desc_info in descriptions['descriptions']:
            desc_id = desc_info['desc_id']
            description = desc_info['description']
            
            # 获取affordance信息并增强文本
            affordance = descid2affordance.get(desc_id, None)
            if affordance and affordance.strip():
                # 通过重复affordance词提升其权重
                enhanced_text = f"{description} [AFFORDANCE] {affordance} {affordance} {affordance}"
            else:
                enhanced_text = description
            
            # 获取文本嵌入
            text_embeddings = get_text_embeddings(enhanced_text, processor, model, device)
            
            # 找到最相似的图片
            similar_images = find_most_similar_images(text_embeddings, image_embeddings, image_names, top_k=10)
            
            # 提取图片名称列表
            top_frame_names = [item['image_name'] for item in similar_images]
            
            # 保存结果
            result = {
                'desc_id': desc_id,
                'description': description,
                'affordance': affordance,
                'image_name': top_frame_names
            }
            
            video_results.append(result)
        
        # 保存到JSON文件
        output_file = os.path.join(visit_output_dir, f"{video_id}_clip4_result.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(video_results, f, ensure_ascii=False, indent=2)
        
        print(f"    结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='CLIP文本-图像相似度匹配')
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录路径')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'], help='数据集分割')
    parser.add_argument('--output_dir', type=str, default='./clip4withaffordance_output', help='输出目录')
    parser.add_argument('--top_k', type=int, default=10, help='返回前k个最相似的图片')
    
    args = parser.parse_args()
    
    print("=== CLIP 文本-图像相似度匹配系统 ===")
    print(f"数据根目录: {args.data_root}")
    print(f"数据集分割: {args.split}")
    print(f"输出目录: {args.output_dir}")
    print(f"返回前{args.top_k}个最相似的图片")
    print()
    
    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 加载模型
        print("正在加载CLIP模型...")
        model_name = "openai/clip-vit-base-patch32"
        
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        
        # 将模型移到指定设备
        model = model.to(device)
        
        print("模型加载成功！")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print()
        
        # 加载数据集信息
        print("正在加载数据集信息...")
        dataset_info = load_dataset_info(args.data_root, args.split)
        print(f"找到 {len(dataset_info)} 个视频")
        
        # 按visit_id分组
        visit_groups = defaultdict(list)
        for item in dataset_info:
            visit_groups[item['visit_id']].append(item['video_id'])
        
        print(f"包含 {len(visit_groups)} 个visit_id")
        print()
        
        # 处理每个visit_id
        start_time = time.time()
        
        for visit_id, video_ids in tqdm(visit_groups.items(), desc="处理visit_id", unit="visit"):
            # 加载描述信息
            descriptions = load_descriptions(args.data_root, args.split, visit_id)
            
            # 处理该visit_id
            process_visit_id(
                visit_id, video_ids, descriptions, 
                args.data_root, args.split, 
                processor, model, device, args.output_dir
            )
        
        end_time = time.time()
        print(f"\n处理完成！")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        print(f"结果已保存到: {args.output_dir}/{args.split}/")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()