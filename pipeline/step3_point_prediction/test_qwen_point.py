import os
import json
import argparse
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
import multiprocessing


# CUDA_VISIBLE_DEVICES=1 python test_qwen_point.py --data_root data/raw_data --split val --clip_root clip4_output

def load_model_and_processor():
    """
    加载Qwen2.5-VL模型和处理器
    
    Returns:
        tuple: (model, processor) 模型和处理器对象
    """
    print("正在加载模型...")
    
    # 加载模型 
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map="auto"
    )
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
    
    print("模型加载完成！")
    return model, processor

def create_messages(image_path, text_prompt):
    """
    创建消息格式
    
    Args:
        image_path (str): 图片路径
        text_prompt (str): 文本提示
        
    Returns:
        list: 消息列表
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    return messages

def validate_and_refine_coordinates(model, processor, image_path, initial_coords, action_description):
    """
    验证和优化初始坐标，确保指向真正的可操作功能部件
    
    Args:
        model: 加载的模型
        processor: 加载的处理器
        image_path (str): 图片路径
        initial_coords (dict): 初始坐标 {"x": x, "y": y}
        action_description (str): 动作描述
        
    Returns:
        dict: 优化后的坐标
    """
    if not initial_coords:
        return initial_coords
    
    # 构建验证提示词
    validation_prompt = f"""Please carefully verify if the following coordinates point to the correct operable functional component.

Action description: "{action_description}"
Current coordinates: ({initial_coords['x']}, {initial_coords['y']})

Please conduct a detailed analysis:

1. **Carefully observe the coordinate location**:
   - Carefully examine the specific location of coordinates ({initial_coords['x']}, {initial_coords['y']}) in the image
   - Analyze whether this location actually contains an operable functional component
   - Consider whether this location is easily accessible for operation

2. **Verify operability**:
   - Confirm that the coordinates point to a genuine operable functional component (such as door handle, switch button, plug, etc.)
   - Check whether this location can actually be operated
   - Avoid pointing to decorative elements, background objects, or non-operable parts

3. **Consider alternative locations**:
   - If the current coordinates are not precise enough, provide more precise coordinates
   - Consider if there are better operable point locations
   - Ensure new coordinates point to the most direct and commonly used operable point

4. **Output result**:
   - If current coordinates are correct, output: "Coordinates are correct"
   - If adjustment is needed, output new coordinates in format: (x, y)

Important notes:
- Please carefully analyze every detail in the image
- Ensure coordinates point to actual operable functional components
- Consider object recognition under different angles and lighting conditions
- Prioritize the most direct and commonly used operable points

Please carefully analyze and output the result:"""

    # 创建消息
    messages = create_messages(image_path, validation_prompt)
    
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # 推理生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    validation_text = output_text[0] if output_text else ""
    
    # 解析验证结果
    import re
    coord_match = re.search(r'\((\d+),\s*(\d+)\)', validation_text)
    if coord_match:
        # 如果模型建议了新的坐标，使用新坐标
        x, y = int(coord_match.group(1)), int(coord_match.group(2))
        return {"x": x, "y": y}
    else:
        # 如果模型认为当前坐标正确，保持原坐标
        return initial_coords

def load_affordance_results(affordance_root, split, visit_id):
    """
    加载affordance结果文件
    
    Args:
        affordance_root (str): affordance结果根目录
        split (str): 数据集分割
        visit_id (str): 访问ID
        
    Returns:
        dict: affordance结果映射 {desc_id: affordance}
    """
    affordance_file = os.path.join(affordance_root, split, f"{visit_id}_affordance.json")
    if not os.path.exists(affordance_file):
        print(f"警告: affordance结果文件不存在: {affordance_file}")
        return {}
    
    try:
        with open(affordance_file, 'r', encoding='utf-8') as f:
            affordance_data = json.load(f)
        
        # 创建desc_id到affordance的映射
        affordance_map = {}
        for item in affordance_data:
            desc_id = item.get("desc_id")
            affordance = item.get("affordance")
            if desc_id and affordance:
                affordance_map[desc_id] = affordance
        
        print(f"加载了 {len(affordance_map)} 个affordance映射")
        return affordance_map
        
    except Exception as e:
        print(f"加载affordance文件时出错: {e}")
        return {}

def predict_coordinates_for_image(model, processor, image_path, action_description, affordance_info=None, max_new_tokens=512, enable_validation=True):
    """
    对单张图片预测坐标
    
    Args:
        model: 加载的模型
        processor: 加载的处理器
        image_path (str): 图片路径
        action_description (str): 动作描述
        affordance_info (str): 可操作物体信息
        max_new_tokens (int): 最大生成token数
        enable_validation (bool): 是否启用坐标验证
        
    Returns:
        dict: 包含坐标信息的字典
    """
    # 构建更精确的坐标预测提示词
    if affordance_info:
        # 如果有affordance信息，在prompt中明确指定要找的功能部件
        coordinate_prompt = f"""You are an extremely precise visual localization assistant. Please carefully and comprehensively analyze the image content to find the specific operable functional component that needs to be operated.

Action description: "{action_description}"
Target functional component: "{affordance_info}"

Please follow these detailed steps for analysis:

1. **Comprehensive image observation**:
   - Carefully observe the overall layout and scene of the image
   - Identify all visible objects and devices in the image
   - Pay attention to image details, including object positions, sizes, shapes, etc.

2. **Identify the target object**:
   - Based on the action description, determine the specific object that needs to be operated
   - The target functional component is specifically: "{affordance_info}"
   - Focus your search on this exact functional component

3. **Precisely locate the target functional component**:
   - You are looking for: "{affordance_info}"
   - This could be:
     * Door handle, door lock, door latch (for doors)
     * Cabinet door handle, drawer handle (for cabinets)
     * Switch button, switch panel, switch lever (for switches)
     * Plug body, socket hole, power interface (for plugs)
     * Button center, button edge (for buttons)
     * Faucet switch, faucet handle (for faucets)
     * Drawer handle, drawer pull ring (for drawers)
     * Window handle, lock catch (for windows)
     * Power button, control panel, display screen (for appliances)
     * Pull handle, hinge, lock (for furniture)

4. **Detailed coordinate position analysis**:
   - Carefully estimate the precise pixel coordinates of the "{affordance_info}" in the image
   - Consider the image dimensions and proportions
   - Ensure coordinates point to the actual "{affordance_info}" component
   - Avoid pointing to decorative elements or approximate positions of entire objects

5. **Output format**:
   - If the "{affordance_info}" is found, output: (x, y)
   - If the target object exists but the specific "{affordance_info}" cannot be determined, output: "Cannot determine coordinates"
   - If the target object does not exist, output: "Object not found"

Important notes:
- You are specifically looking for: "{affordance_info}"
- Please carefully analyze every detail in the image
- Coordinates should be precise pixel coordinates on the image
- Prioritize the most direct and commonly used operable functional components
- Consider object recognition under different angles and lighting conditions
- CRITICAL: Always target the specific functional component "{affordance_info}", not just the general object

Please carefully analyze the image and output coordinates:"""
    else:
        # 如果没有affordance信息，使用原来的通用prompt
        coordinate_prompt = f"""You are an extremely precise visual localization assistant. Please carefully and comprehensively analyze the image content to find the specific operable functional component that needs to be operated.

Action description: "{action_description}"

Please follow these detailed steps for analysis:

1. **Comprehensive image observation**:
   - Carefully observe the overall layout and scene of the image
   - Identify all visible objects and devices in the image
   - Pay attention to image details, including object positions, sizes, shapes, etc.

2. **Identify the target object**:
   - Based on the action description, determine the specific object that needs to be operated
   - Consider synonyms and similar expressions (e.g., "switch" might refer to "button", "panel", etc.)
   - If the object is not obvious, consider possible alternative objects

3. **Precisely locate the operable functional component**:
   - Find the specific operable functional component on the target object, for example:
     * Door → door handle, door lock, door latch
     * Cabinet → cabinet door handle, drawer handle, cabinet door edge
     * Switch → switch button, switch panel, switch lever
     * Plug → plug body, socket hole, power interface
     * Button → button center, button edge
     * Faucet → faucet switch, faucet handle
     * Drawer → drawer handle, drawer pull ring
     * Window → window handle, lock catch, window frame
     * Appliance → power button, control panel, display screen
     * Furniture → pull handle, hinge, lock

4. **Detailed coordinate position analysis**:
   - Carefully estimate the precise pixel coordinates of the operable functional component in the image
   - Consider the image dimensions and proportions
   - Ensure coordinates point to actual operable functional components
   - Avoid pointing to decorative elements or approximate positions of entire objects

5. **Output format**:
   - If operable functional component is found, output: (x, y)
   - If object exists but specific operable point cannot be determined, output: "Cannot determine coordinates"
   - If object does not exist, output: "Object not found"

Important notes:
- Please carefully analyze every detail in the image
- Coordinates should be precise pixel coordinates on the image
- Prioritize the most direct and commonly used operable functional components
- If an object has multiple operable points, choose the one that best matches the action description
- Consider object recognition under different angles and lighting conditions
- CRITICAL: Always target the specific functional component (e.g., "door handle" for "open bedroom door", not just the door edge)

Please carefully analyze the image and output coordinates:"""
    
    # 创建消息
    messages = create_messages(image_path, coordinate_prompt)
    
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # 推理生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    result_text = output_text[0] if output_text else ""
    
    # 解析结果
    result = {
        "image_name": os.path.basename(image_path),
        "raw_response": result_text,
        "coordinates": None,
        "object_found": False,
        "affordance_info": affordance_info
    }
    
    # 尝试从输出中提取坐标
    try:
        # 首先检查是否返回"Object not found"
        if "object not found" in result_text.lower():
            result["coordinates"] = None
            result["object_found"] = False
        else:
            # 查找坐标格式 (x, y)
            import re
            coord_match = re.search(r'\((\d+),\s*(\d+)\)', result_text)
            if coord_match:
                x, y = int(coord_match.group(1)), int(coord_match.group(2))
                initial_coords = {"x": x, "y": y}
                
                # 如果启用验证，进行坐标验证和优化
                if enable_validation:
                    try:
                        refined_coords = validate_and_refine_coordinates(
                            model, processor, image_path, initial_coords, action_description
                        )
                        result["coordinates"] = refined_coords
                        result["object_found"] = True
                    except Exception as e:
                        print(f"坐标验证失败，使用初始坐标: {e}")
                        result["coordinates"] = initial_coords
                        result["object_found"] = True
                else:
                    result["coordinates"] = initial_coords
                    result["object_found"] = True
            else:
                result["coordinates"] = None
                result["object_found"] = False
    except:
        result["coordinates"] = None
        result["object_found"] = False
    
    return result

def predict_coordinates_second_attempt(model, processor, image_path, action_description, affordance_info, max_new_tokens=512, enable_validation=True):
    """
    第二次尝试预测坐标 - 使用更直接的询问方式
    
    Args:
        model: 加载的模型
        processor: 加载的处理器
        image_path (str): 图片路径
        action_description (str): 动作描述
        affordance_info (str): 可操作物体信息
        max_new_tokens (int): 最大生成token数
        enable_validation (bool): 是否启用坐标验证
        
    Returns:
        dict: 包含坐标信息的字典
    """
    # 构建第二次尝试的提示词
    if affordance_info:
        second_attempt_prompt = f"""Please look at this image carefully and answer my question.

Question: Do you see a "{affordance_info}" in this image?

If you see the "{affordance_info}", please:
1. Confirm that you can see it
2. Point to its exact location in the image by providing the pixel coordinates

Please respond in this format:
- If you see the "{affordance_info}": "Yes, I can see the {affordance_info} at coordinates (x, y)"
- If you don't see the "{affordance_info}": "No, I cannot see the {affordance_info} in this image"

Important:
- Be very precise about the location
- Provide exact pixel coordinates where the {affordance_info} is located
- If you're unsure about the exact position, say so

Please answer:"""
    else:
        # 如果没有affordance信息，使用通用询问
        second_attempt_prompt = f"""Please look at this image carefully and answer my question.

Question: Do you see any operable functional component related to "{action_description}" in this image?

If you see any relevant operable component, please:
1. Confirm that you can see it
2. Point to its exact location in the image by providing the pixel coordinates

Please respond in this format:
- If you see a relevant component: "Yes, I can see [component name] at coordinates (x, y)"
- If you don't see any relevant component: "No, I cannot see any relevant operable component in this image"

Important:
- Be very precise about the location
- Provide exact pixel coordinates where the component is located
- If you're unsure about the exact position, say so

Please answer:"""
    
    # 创建消息
    messages = create_messages(image_path, second_attempt_prompt)
    
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # 推理生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    result_text = output_text[0] if output_text else ""
    
    # 解析结果
    result = {
        "image_name": os.path.basename(image_path),
        "raw_response": result_text,
        "coordinates": None,
        "object_found": False,
        "affordance_info": affordance_info,
        "is_second_attempt": True
    }
    
    # 尝试从输出中提取坐标
    try:
        # 检查是否包含"yes"或"can see"等肯定回答
        if any(keyword in result_text.lower() for keyword in ["yes", "can see", "i can see"]):
            # 查找坐标格式 (x, y)
            import re
            coord_match = re.search(r'\((\d+),\s*(\d+)\)', result_text)
            if coord_match:
                x, y = int(coord_match.group(1)), int(coord_match.group(2))
                initial_coords = {"x": x, "y": y}
                
                # 如果启用验证，进行坐标验证和优化
                if enable_validation:
                    try:
                        refined_coords = validate_and_refine_coordinates(
                            model, processor, image_path, initial_coords, action_description
                        )
                        result["coordinates"] = refined_coords
                        result["object_found"] = True
                    except Exception as e:
                        print(f"第二次尝试坐标验证失败，使用初始坐标: {e}")
                        result["coordinates"] = initial_coords
                        result["object_found"] = True
                else:
                    result["coordinates"] = initial_coords
                    result["object_found"] = True
            else:
                result["coordinates"] = None
                result["object_found"] = False
        else:
            result["coordinates"] = None
            result["object_found"] = False
    except:
        result["coordinates"] = None
        result["object_found"] = False
    
    return result

def predict_fallback_possible_point(model, processor, image_path, action_description, max_new_tokens=512, enable_validation=True):
    """
    在所有图片都未找到affordance点时，预测最可能的可操作点
    """
    fallback_prompt = f"""You are an extremely intelligent visual assistant. In the following image, although you could not find the exact operable functional component (such as a handle, button, or switch) required by the action, please carefully analyze the image and output the most likely location that a person would try to operate in order to complete the action described below.\n\nAction description: \"{action_description}\"\n\nInstructions:\n- If you cannot find the exact target, please output the most likely operable point based on the scene and action.\n- Output the coordinates in the format: (x, y)\n- If you really cannot determine any possible point, output: \"Cannot determine coordinates\"\n\nPlease analyze the image and output the most likely coordinates:"""

    messages = create_messages(image_path, fallback_prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    result_text = output_text[0] if output_text else ""
    import re
    coord_match = re.search(r'\((\d+),\s*(\d+)\)', result_text)
    result = {
        "image_name": os.path.basename(image_path),
        "raw_response": result_text,
        "coordinates": None,
        "object_found": False,
        "affordance_info": None,
        "is_fallback": True
    }
    if coord_match:
        x, y = int(coord_match.group(1)), int(coord_match.group(2))
        initial_coords = {"x": x, "y": y}
        if enable_validation:
            try:
                refined_coords = validate_and_refine_coordinates(
                    model, processor, image_path, initial_coords, action_description
                )
                result["coordinates"] = refined_coords
                result["object_found"] = True
            except Exception as e:
                print(f"Fallback坐标验证失败，使用初始坐标: {e}")
                result["coordinates"] = initial_coords
                result["object_found"] = True
        else:
            result["coordinates"] = initial_coords
            result["object_found"] = True
    return result

def load_descriptions(data_root, split, visit_id):
    """
    加载描述文件
    
    Args:
        data_root (str): 数据根目录
        split (str): 数据集分割 (train/val)
        visit_id (str): 访问ID
        
    Returns:
        dict: 描述数据
    """
    desc_file = os.path.join(data_root, split, visit_id, f"{visit_id}_descriptions.json")
    if not os.path.exists(desc_file):
        print(f"警告: 描述文件不存在: {desc_file}")
        return None
    
    with open(desc_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_clip4_results(clip_root, split, visit_id, video_id):
    """
    加载CLIP4结果文件
    Args:
        clip_root (str): clip结果根目录
        split (str): 数据集分割
        visit_id (str): 访问ID
        video_id (str): 视频ID
    Returns:
        list: CLIP4结果列表
    """
    # 根据clip_root内容判断文件名
    if 'clip4' in os.path.basename(clip_root):
        json_file = f"{video_id}_clip4_result.json"
    else:
        json_file = f"{video_id}_result.json"
    clip4_file = os.path.join(clip_root, split, visit_id, json_file)
    if not os.path.exists(clip4_file):
        print(f"警告: CLIP4结果文件不存在: {clip4_file}")
        return None
    with open(clip4_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_single_video(model, processor, data_root, split, visit_id, video_id, clip_root, affordance_map=None, enable_validation=True):
    """
    处理单个视频的所有描述和帧
    
    Args:
        model: 加载的模型
        processor: 加载的处理器
        data_root (str): 数据根目录
        split (str): 数据集分割
        visit_id (str): 访问ID
        video_id (str): 视频ID
        clip_root (str): clip结果根目录
        affordance_map (dict): affordance映射
        enable_validation (bool): 是否启用坐标验证
        
    Returns:
        list: 处理结果列表
    """
    print(f"\n处理视频: {visit_id}/{video_id}")
    
    # 加载描述文件
    desc_data = load_descriptions(data_root, split, visit_id)
    if desc_data is None:
        return []
    
    # 加载CLIP4结果
    clip4_results = load_clip4_results(clip_root, split, visit_id, video_id)
    if clip4_results is None:
        return []
    
    # 创建描述ID到描述的映射
    desc_map = {desc["desc_id"]: desc["description"] for desc in desc_data["descriptions"]}
    
    results = []
    
    # 处理每个CLIP4结果
    for clip4_item in clip4_results:
        desc_id = clip4_item["desc_id"]
        description = desc_map.get(desc_id, clip4_item["description"])
        top_frames = clip4_item["image_name"]
        
        # 获取对应的affordance信息
        affordance_info = affordance_map.get(desc_id) if affordance_map else None
        
        print(f"  处理 desc_id: {desc_id}")
        if affordance_info:
            print(f"    目标功能部件: {affordance_info}")
        
        # 为每个top_frame预测坐标
        frame_results = []
        first_attempt_found = False
        
        # 第一次尝试：使用标准方法
        print(f"    🔍 第一次尝试：标准坐标预测")
        for frame_name in top_frames:
            # 构建图片路径
            image_path = os.path.join(data_root, split, visit_id, video_id, "hires_wide", frame_name)
            
            # 检查图片是否存在
            if not os.path.exists(image_path):
                print(f"警告: 图片不存在: {image_path}")
                continue
            
            # 预测坐标
            try:
                coord_result = predict_coordinates_for_image(
                    model, processor, image_path, description, affordance_info, enable_validation=enable_validation
                )
                frame_results.append(coord_result)
                
                # 检查是否找到了可操作点
                if coord_result.get("object_found") and coord_result.get("coordinates"):
                    first_attempt_found = True
                    print(f"    ✅ 在帧 {frame_name} 中找到可操作点")
                
            except Exception as e:
                print(f"处理图片时出错 {image_path}: {e}")
                frame_results.append({
                    "image_name": frame_name,
                    "raw_response": f"Error: {str(e)}",
                    "coordinates": None,
                    "object_found": False,
                    "affordance_info": affordance_info,
                    "is_second_attempt": False
                })
        
        # 如果第一次尝试没有找到可操作点，进行第二次尝试
        if not first_attempt_found:
            print(f"    ⚠️  第一次尝试未找到可操作点，启动第二次尝试：直接询问")
            
            for frame_name in top_frames:
                # 构建图片路径
                image_path = os.path.join(data_root, split, visit_id, video_id, "hires_wide", frame_name)
                
                # 检查图片是否存在
                if not os.path.exists(image_path):
                    continue
                
                # 第二次尝试：使用直接询问方式
                try:
                    coord_result = predict_coordinates_second_attempt(
                        model, processor, image_path, description, affordance_info, enable_validation=enable_validation
                    )
                    
                    # 将第二次尝试的结果添加到frame_results中
                    # 找到对应的第一次尝试结果并更新
                    for i, existing_result in enumerate(frame_results):
                        if existing_result["image_name"] == frame_name:
                            # 如果第二次尝试找到了坐标，更新结果
                            if coord_result.get("object_found") and coord_result.get("coordinates"):
                                frame_results[i] = coord_result
                                print(f"    ✅ 第二次尝试在帧 {frame_name} 中找到可操作点")
                            break
                    else:
                        # 如果没有找到对应的第一次尝试结果，直接添加
                        frame_results.append(coord_result)
                        if coord_result.get("object_found") and coord_result.get("coordinates"):
                            print(f"    ✅ 第二次尝试在帧 {frame_name} 中找到可操作点")
                
                except Exception as e:
                    print(f"第二次尝试处理图片时出错 {image_path}: {e}")
                    # 添加错误结果
                    error_result = {
                        "image_name": frame_name,
                        "raw_response": f"Second attempt error: {str(e)}",
                        "coordinates": None,
                        "object_found": False,
                        "affordance_info": affordance_info,
                        "is_second_attempt": True
                    }
                    frame_results.append(error_result)
        
        # 如果第一次和第二次尝试都没有找到可操作点，进行fallback最可能点推理
        found_any = any(f.get("object_found") for f in frame_results)
        if not found_any and len(top_frames) > 0:
            print(f"    ⚠️  所有图片都未找到affordance点，尝试对所有图片输出最可能的操作点（fallback）")
            for fallback_frame in top_frames:
                fallback_image = os.path.join(data_root, split, visit_id, video_id, "hires_wide", fallback_frame)
                if os.path.exists(fallback_image):
                    fallback_result = predict_fallback_possible_point(
                        model, processor, fallback_image, description, enable_validation=enable_validation
                    )
                    frame_results.append(fallback_result)
                    print(f"    ✅ fallback最可能点已添加: {fallback_result.get('coordinates')} for {fallback_frame}")
                else:
                    print(f"    ⚠️ fallback图片不存在: {fallback_image}")
        
        # 添加到结果中
        results.append({
            "desc_id": desc_id,
            "description": description,
            "affordance_info": affordance_info,
            "frame_results": frame_results,
            "found_operable_point": any(f.get("object_found") for f in frame_results),
            "used_second_attempt": not first_attempt_found and any(f.get("is_second_attempt") and f.get("object_found") for f in frame_results)
        })
    
    return results

def save_results(results, split, visit_id, video_id, output_root):
    """
    保存结果到JSON文件
    
    Args:
        results (list): 处理结果
        split (str): 数据集分割
        visit_id (str): 访问ID
        video_id (str): 视频ID
        output_root (str): 输出json根目录
    """
    # 创建输出目录
    output_dir = os.path.join(output_root, visit_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    output_file = os.path.join(output_dir, f"{video_id}_point.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_file}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='批量坐标预测')
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录路径')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'], help='数据集分割')
    parser.add_argument('--clip_root', type=str, default='clip4_output', help='clip结果根目录')
    parser.add_argument('--affordance_root', type=str, default='qwen/affordance_result', help='affordance结果根目录')
    parser.add_argument('--output_root', type=str, help='输出json根目录（可选，默认自动生成）')
    parser.add_argument('--enable_validation', action='store_true', default=True, help='是否启用坐标验证和优化')
    parser.add_argument('--disable_validation', dest='enable_validation', action='store_false', help='禁用坐标验证和优化')
    args = parser.parse_args()
    
    print(f"数据根目录: {args.data_root}")
    print(f"数据集分割: {args.split}")
    print(f"clip结果根目录: {args.clip_root}")
    print(f"affordance结果根目录: {args.affordance_root}")
    print(f"坐标验证: {'启用' if args.enable_validation else '禁用'}")
    
    # 设置输出目录
    if args.output_root:
        output_root = args.output_root
        print(f"输出json根目录: {output_root}")
    else:
        # 自动生成输出目录
        # 取clip_root中'_output'前面的部分作为保存路径的一部分
        clip_root_base = os.path.basename(args.clip_root)
        if '_output' in clip_root_base:
            clipr4_part = clip_root_base.split('_output')[0]
            output_dir_name = f"point_{clipr4_part}_output"
        else:
            clipr4_part = clip_root_base
            output_dir_name = f"point_{clipr4_part}_output"
        # 输出路径始终挂在/data/helian/affseg/qwen2下
        output_root = os.path.join("/data/helian/affseg/qwen2", output_dir_name, args.split)
        print(f"输出json根目录: {output_root} (自动生成)")
    
    # 加载模型
    model, processor = load_model_and_processor()
    
    # 获取所有visit_id和video_id
    data_dir = os.path.join(args.data_root, args.split)
    visit_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for visit_id in tqdm(visit_ids, desc="处理visit_id进度"):
        print(f"\n{'='*60}")
        print(f"处理 visit_id: {visit_id}")
        print(f"{'='*60}")
        affordance_map = load_affordance_results(args.affordance_root, args.split, visit_id)
        visit_dir = os.path.join(data_dir, visit_id)
        video_ids = [d for d in os.listdir(visit_dir) if os.path.isdir(os.path.join(visit_dir, d))]
        print(f"找到 {len(video_ids)} 个video_id: {video_ids}")
        for video_id in video_ids:
            output_dir = os.path.join(output_root, visit_id)
            output_file = os.path.join(output_dir, f"{video_id}_point.json")
            if os.path.exists(output_file):
                print(f"  跳过 video_id: {video_id}（已处理）")
                continue
            print(f"  处理 video_id: {video_id}")
            try:
                results = process_single_video(
                    model, processor, args.data_root, args.split, visit_id, video_id,
                    args.clip_root, affordance_map, args.enable_validation
                )
                if results:
                    save_results(results, args.split, visit_id, video_id, output_root)
                else:
                    print(f"跳过 {video_id}: 没有有效结果")
            except Exception as e:
                print(f"处理 {visit_id}/{video_id} 时出错: {e}")
                continue

if __name__ == "__main__":
    main() 