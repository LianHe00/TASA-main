import os
import re
import sys
import time
import json
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, SamModel, SamProcessor

# python test_molmo.py

def init_sam_model(device: str):
    print("Loading SAM model...")
    start_time = time.time()
    device = torch.device(device)
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    print(f"SAM model loaded in {time.time() - start_time:.2f} seconds")
    return model, processor

def init_molmo_model(device: str):
    print("Loading Molmo processor...")
    start_time = time.time()
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    print(f"Processor loaded in {time.time() - start_time:.2f} seconds")
    print("Loading Molmo model...")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto',
        low_cpu_mem_usage=True
    )
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    print(f"Model is on device: {next(model.parameters()).device}")
    return model, processor

def extract_points(molmo_output: str, size: tuple) -> np.array:
    image_w, image_h = size
    all_points = []
    print(f"Molmo输出: {molmo_output}")
    print(f"图像尺寸: {size}")
    for match in re.finditer(
        r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"',
        molmo_output,
    ):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
            print(f"找到点: {point}")
        except ValueError:
            print(f"无法解析点: {match.groups()}")
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                print(f"点超出范围，跳过: {point}")
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            print(f"转换后点: {point}")
            all_points.append(point)
    if len(all_points) > 0:
        points = np.stack(all_points, axis=0)
        print(f"最终点坐标: {points}")
    else:
        points = None
        print("未找到有效点")
    return points

def process_sam_prompts(sam_model, sam_processor, img, points):
    if points is None or len(points) == 0:
        return None, None
    try:
        input_points = points.tolist()
        input_labels = [[1] for _ in range(len(input_points))]
        print(f"处理 {len(input_points)} 个点: {input_points}")
        inputs = sam_processor(
            img,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(sam_model.device)
        image_embeddings = sam_model.get_image_embeddings(inputs["pixel_values"])
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})
        inputs["input_points"] = inputs["input_points"].unsqueeze(0)
        with torch.no_grad():
            outputs = sam_model(**inputs)
    except Exception as e:
        print(f"SAM模型推理失败: {e}")
        return None, None
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    scores = outputs.iou_scores[0]
    masks = masks.cpu()
    scores = scores.cpu()
    top_scores_idxs = torch.argmax(scores, dim=1)
    idxs = torch.arange(0, masks.shape[0])
    try:
        masks = masks[idxs, top_scores_idxs]
        selected_scores = scores[idxs, top_scores_idxs]
    except Exception as e:
        print(f"SAM掩码选择失败: {e}")
        print(f"masks shape: {masks.shape}")
        print(f"scores shape: {scores.shape}")
        print(f"idxs shape: {idxs.shape}")
        print(f"top_scores_idxs shape: {top_scores_idxs.shape}")
        return None, None
    return masks.numpy(), selected_scores.numpy()

def save_mask_overlay(image_path, masks, points, scores, output_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    for i, mask in enumerate(masks):
        colored_mask = np.zeros_like(img_rgb)
        colored_mask[mask] = [255, 0, 0]
        overlay = img_rgb.copy()
        overlay[mask] = overlay[mask] * 0.5 + colored_mask[mask] * 0.5
        overlay_pil = Image.fromarray(overlay)
        draw_overlay = ImageDraw.Draw(overlay_pil)
        if points is not None and len(points) > 0:
            for j, point in enumerate(points):
                pixel_x, pixel_y = int(point[0]), int(point[1])
                circle_radius = 8
                draw_overlay.ellipse(
                    [pixel_x - circle_radius, pixel_y - circle_radius,
                     pixel_x + circle_radius, pixel_y + circle_radius],
                    fill='red',
                    outline='white',
                    width=2
                )
                line_length = 15
                draw_overlay.line([pixel_x - line_length, pixel_y, pixel_x + line_length, pixel_y],
                                 fill='white', width=2)
                draw_overlay.line([pixel_x, pixel_y - line_length, pixel_x, pixel_y + line_length],
                                 fill='white', width=2)
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                text_x = pixel_x + 15
                text_y = pixel_y - 25
                if font:
                    text_bbox = draw_overlay.textbbox((text_x, text_y), f"P{j+1}", font=font)
                    draw_overlay.rectangle(text_bbox, fill='black', outline='white', width=1)
                    draw_overlay.text((text_x, text_y), f"P{j+1}", fill='white', font=font)
                else:
                    draw_overlay.text((text_x, text_y), f"P{j+1}", fill='white')
        if scores is not None and len(scores) > i:
            confidence = scores[i].item() if hasattr(scores[i], 'item') else scores[i]
            confidence_text = f"SAM Confidence: {confidence:.3f}"
            try:
                font = ImageFont.load_default()
            except:
                font = None
            if font:
                text_bbox = draw_overlay.textbbox((10, 10), confidence_text, font=font)
                draw_overlay.rectangle(text_bbox, fill='black', outline='white', width=2)
                draw_overlay.text((10, 10), confidence_text, fill='white', font=font)
            else:
                draw_overlay.text((10, 10), confidence_text, fill='white')
        overlay_cv = cv2.cvtColor(np.array(overlay_pil), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, overlay_cv)

def save_mask_data(masks, points, output_path):
    if masks is not None:
        orig_dims = np.asarray([m.shape for m in masks])
        resized_masks = []
        for mask in masks:
            if mask.shape != (1920, 1440):
                mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(torch.float)
                mask_resized = torch.nn.functional.interpolate(
                    mask_tensor, (1920, 1440), mode="nearest"
                ).squeeze().numpy().astype(np.uint8)
                resized_masks.append(mask_resized)
            else:
                resized_masks.append(mask.astype(np.uint8))
        np.savez_compressed(
            output_path,
            masks=np.stack(resized_masks, axis=0),
            points=points,
            orig_dims=orig_dims,
        )
    else:
        empty = np.asarray([0])
        np.savez_compressed(
            output_path,
            masks=empty,
            points=empty,
            orig_dims=empty,
        )

def process_single_image(molmo_model, molmo_processor, sam_model, sam_processor,
                        image_path, prompt, output_dir, image_name, desc_id=None):
    if desc_id:
        print(f"[{desc_id}] 处理图片: {image_name}")
    else:
        print(f"处理图片: {image_name}")
    image = Image.open(image_path)
    image_size = image.size
    start_time = time.time()
    inputs = molmo_processor.process(images=[image], text=prompt)
    inputs = {k: v.to(molmo_model.device).unsqueeze(0) for k, v in inputs.items()}
    with torch.no_grad():
        output = molmo_model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=molmo_processor.tokenizer
        )
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = molmo_processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    molmo_time = time.time() - start_time
    points = extract_points(generated_text, image_size)
    if points is not None:
        start_time = time.time()
        masks, scores = process_sam_prompts(sam_model, sam_processor, image, points)
        sam_time = time.time() - start_time
        if masks is not None:
            mask_data_path = os.path.join(output_dir, f"{image_name}_mask_data.npz")
            save_mask_data(masks, points, mask_data_path)
            mask_overlay_path = os.path.join(output_dir, f"{image_name}_mask_overlay.jpg")
            save_mask_overlay(image_path, masks, points, scores, mask_overlay_path)
            print(f"  ✓ 成功 - Molmo: {molmo_time:.1f}s, SAM: {sam_time:.1f}s, 点: {len(points)}, 掩码: {len(masks)}")
            if len(masks) > 0:
                mask_area = np.sum(masks[0])
                total_pixels = masks[0].shape[0] * masks[0].shape[1]
                coverage = (mask_area / total_pixels) * 100
                if scores is not None and len(scores) > 0:
                    confidence = scores[0].item() if hasattr(scores[0], 'item') else scores[0]
                    print(f"    掩码覆盖: {coverage:.1f}%, 置信度: {confidence:.3f}")
            return True
        else:
            print(f"  ✗ SAM失败 - Molmo: {molmo_time:.1f}s")
            return False
    else:
        print(f"  ✗ 无有效点 - Molmo: {molmo_time:.1f}s")
        return False

def find_crop_images(root_dir):
    """递归查找所有_crop.jpg图片，返回(图片路径, split, visit_id, video_id, desc_id, 图片名)元组列表"""
    crop_images = []
    # split为val或train
    split = os.path.basename(root_dir.rstrip('/'))
    for visit_id in os.listdir(root_dir):
        visit_path = os.path.join(root_dir, visit_id)
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
                for fname in os.listdir(desc_path):
                    if fname.endswith('_crop.jpg'):
                        crop_images.append((
                            os.path.join(desc_path, fname),
                            split, visit_id, video_id, desc_id, fname
                        ))
    return crop_images

def get_affordance_info(split, visit_id, video_id, desc_id):
    json_path = f"qwen2/affordance/{split}/{visit_id}_affordance.json"
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        for item in data:
            if item.get('desc_id') == desc_id:
                return item.get('affordance')
    except Exception as e:
        print(f"读取affordance失败: {json_path}, {e}")
    return None

def main():
    # 输入根目录
    root_dir = 'seg_image/point_clipwithaffordance_output/val'
    output_root = 'molmo_output'  # 可自定义输出目录
    os.makedirs(output_root, exist_ok=True)

    # 初始化模型
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    molmo_model, molmo_processor = init_molmo_model(device)
    sam_model, sam_processor = init_sam_model(device)

    # 查找所有_crop.jpg图片
    crop_images = find_crop_images(root_dir)
    print(f"共找到 {len(crop_images)} 张_crop.jpg图片")

    for idx, (img_path, split, visit_id, video_id, desc_id, fname) in enumerate(crop_images):
        print(f"\n[{idx+1}/{len(crop_images)}] 处理: {img_path}")
        # 动态获取prompt
        affordance_info = get_affordance_info(split, visit_id, video_id, desc_id)
        if affordance_info and affordance_info.strip():
            prompt = f"point to {affordance_info}"
        else:
            prompt = "point to the affordance."
        # 输出目录结构与输入一致
        out_dir = os.path.join(output_root, visit_id, video_id, desc_id)
        os.makedirs(out_dir, exist_ok=True)
        image_name = os.path.splitext(fname)[0]
        process_single_image(
            molmo_model, molmo_processor, sam_model, sam_processor,
            img_path, prompt, out_dir, image_name, desc_id
        )

if __name__ == "__main__":
    main()
