import os
import json
import argparse
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
import multiprocessing


def load_model_and_processor():
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
    print("Model loaded.")
    return model, processor

def create_messages(image_path, text_prompt):
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
    if not initial_coords:
        return initial_coords
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

    messages = create_messages(image_path, validation_prompt)
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
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    validation_text = output_text[0] if output_text else ""
    import re
    coord_match = re.search(r'\((\d+),\s*(\d+)\)', validation_text)
    if coord_match:
        x, y = int(coord_match.group(1)), int(coord_match.group(2))
        return {"x": x, "y": y}
    else:
        return initial_coords

def load_affordance_results(affordance_root, split, visit_id):
    affordance_file = os.path.join(affordance_root, split, f"{visit_id}_affordance.json")
    if not os.path.exists(affordance_file):
        print(f"Warning: affordance file not found: {affordance_file}")
        return {}
    
    try:
        with open(affordance_file, 'r', encoding='utf-8') as f:
            affordance_data = json.load(f)
        
        affordance_map = {}
        for item in affordance_data:
            desc_id = item.get("desc_id")
            affordance = item.get("affordance")
            if desc_id and affordance:
                affordance_map[desc_id] = affordance
        
        print(f"Loaded {len(affordance_map)} affordance mapping(s)")
        return affordance_map
    except Exception as e:
        print(f"Error loading affordance file: {e}")
        return {}

def predict_coordinates_for_image(model, processor, image_path, action_description, affordance_info=None, max_new_tokens=512, enable_validation=True):
    if affordance_info:
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
    messages = create_messages(image_path, coordinate_prompt)
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
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    result_text = output_text[0] if output_text else ""
    result = {
        "image_name": os.path.basename(image_path),
        "raw_response": result_text,
        "coordinates": None,
        "object_found": False,
        "affordance_info": affordance_info
    }
    try:
        if "object not found" in result_text.lower():
            result["coordinates"] = None
            result["object_found"] = False
        else:
            import re
            coord_match = re.search(r'\((\d+),\s*(\d+)\)', result_text)
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
                        print(f"Coordinate validation failed, using initial: {e}")
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
    messages = create_messages(image_path, second_attempt_prompt)
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
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    result_text = output_text[0] if output_text else ""
    result = {
        "image_name": os.path.basename(image_path),
        "raw_response": result_text,
        "coordinates": None,
        "object_found": False,
        "affordance_info": affordance_info,
        "is_second_attempt": True
    }
    try:
        if any(keyword in result_text.lower() for keyword in ["yes", "can see", "i can see"]):
            import re
            coord_match = re.search(r'\((\d+),\s*(\d+)\)', result_text)
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
                        print(f"Second attempt validation failed, using initial: {e}")
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
                print(f"Fallback validation failed, using initial: {e}")
                result["coordinates"] = initial_coords
                result["object_found"] = True
        else:
            result["coordinates"] = initial_coords
            result["object_found"] = True
    return result

def load_descriptions(data_root, split, visit_id):
    desc_file = os.path.join(data_root, split, visit_id, f"{visit_id}_descriptions.json")
    if not os.path.exists(desc_file):
        print(f"Warning: description file not found: {desc_file}")
        return None
    
    with open(desc_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_clip4_results(clip_root, split, visit_id, video_id):
    if 'clip4' in os.path.basename(clip_root):
        json_file = f"{video_id}_clip4_result.json"
    else:
        json_file = f"{video_id}_result.json"
    clip4_file = os.path.join(clip_root, split, visit_id, json_file)
    if not os.path.exists(clip4_file):
        print(f"Warning: CLIP4 result file not found: {clip4_file}")
        return None
    with open(clip4_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_single_video(model, processor, data_root, split, visit_id, video_id, clip_root, affordance_map=None, enable_validation=True):
    print(f"\nProcessing video: {visit_id}/{video_id}")

    desc_data = load_descriptions(data_root, split, visit_id)
    if desc_data is None:
        return []

    clip4_results = load_clip4_results(clip_root, split, visit_id, video_id)
    if clip4_results is None:
        return []

    desc_map = {desc["desc_id"]: desc["description"] for desc in desc_data["descriptions"]}

    results = []

    for clip4_item in clip4_results:
        desc_id = clip4_item["desc_id"]
        description = desc_map.get(desc_id, clip4_item["description"])
        top_frames = clip4_item["image_name"]

        affordance_info = affordance_map.get(desc_id) if affordance_map else None

        print(f"  desc_id: {desc_id}")
        if affordance_info:
            print(f"    affordance: {affordance_info}")

        frame_results = []
        first_attempt_found = False
        
        print(f"    First attempt: standard coordinate prediction")
        for frame_name in top_frames:
            image_path = os.path.join(data_root, split, visit_id, video_id, "hires_wide", frame_name)
            if not os.path.exists(image_path):
                print(f"Warning: image not found: {image_path}")
                continue
            try:
                coord_result = predict_coordinates_for_image(
                    model, processor, image_path, description, affordance_info, enable_validation=enable_validation
                )
                frame_results.append(coord_result)
                if coord_result.get("object_found") and coord_result.get("coordinates"):
                    first_attempt_found = True
                    print(f"    Found operable point in frame {frame_name}")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                frame_results.append({
                    "image_name": frame_name,
                    "raw_response": f"Error: {str(e)}",
                    "coordinates": None,
                    "object_found": False,
                    "affordance_info": affordance_info,
                    "is_second_attempt": False
                })
        
        if not first_attempt_found:
            print(f"    First attempt found no point, second attempt: direct query")
            for frame_name in top_frames:
                image_path = os.path.join(data_root, split, visit_id, video_id, "hires_wide", frame_name)
                if not os.path.exists(image_path):
                    continue
                try:
                    coord_result = predict_coordinates_second_attempt(
                        model, processor, image_path, description, affordance_info, enable_validation=enable_validation
                    )
                    for i, existing_result in enumerate(frame_results):
                        if existing_result["image_name"] == frame_name:
                            if coord_result.get("object_found") and coord_result.get("coordinates"):
                                frame_results[i] = coord_result
                                print(f"    Second attempt found point in frame {frame_name}")
                            break
                    else:
                        frame_results.append(coord_result)
                        if coord_result.get("object_found") and coord_result.get("coordinates"):
                            print(f"    Second attempt found point in frame {frame_name}")
                except Exception as e:
                    print(f"Second attempt error {image_path}: {e}")
                    error_result = {
                        "image_name": frame_name,
                        "raw_response": f"Second attempt error: {str(e)}",
                        "coordinates": None,
                        "object_found": False,
                        "affordance_info": affordance_info,
                        "is_second_attempt": True
                    }
                    frame_results.append(error_result)
        found_any = any(f.get("object_found") for f in frame_results)
        if not found_any and len(top_frames) > 0:
            print(f"    No affordance point found, trying fallback")
            for fallback_frame in top_frames:
                fallback_image = os.path.join(data_root, split, visit_id, video_id, "hires_wide", fallback_frame)
                if os.path.exists(fallback_image):
                    fallback_result = predict_fallback_possible_point(
                        model, processor, fallback_image, description, enable_validation=enable_validation
                    )
                    frame_results.append(fallback_result)
                    print(f"    Fallback point added: {fallback_result.get('coordinates')} for {fallback_frame}")
                else:
                    print(f"    Fallback image not found: {fallback_image}")
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
    output_dir = os.path.join(output_root, visit_id)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{video_id}_point.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Batch point prediction')
    parser.add_argument('--data_root', type=str, required=True, help='Path to data root')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'], help='Dataset split')
    parser.add_argument('--clip_root', type=str, default='path/to/clip4_output', help='Path to CLIP results')
    parser.add_argument('--affordance_root', type=str, default='path/to/affordance_result', help='Path to affordance results')
    parser.add_argument('--output_root', type=str, default=None, help='Output root (default: path/to/qwen2_output/...)')
    parser.add_argument('--enable_validation', action='store_true', default=True, help='Enable coordinate validation')
    parser.add_argument('--disable_validation', dest='enable_validation', action='store_false', help='Disable coordinate validation')
    args = parser.parse_args()

    print(f"Data root: {args.data_root}, split: {args.split}")
    print(f"CLIP root: {args.clip_root}, affordance root: {args.affordance_root}")
    print(f"Validation: {'on' if args.enable_validation else 'off'}")

    if args.output_root:
        output_root = args.output_root
    else:
        clip_root_base = os.path.basename(args.clip_root.rstrip('/'))
        if '_output' in clip_root_base:
            clipr4_part = clip_root_base.split('_output')[0]
        else:
            clipr4_part = clip_root_base
        output_dir_name = f"point_{clipr4_part}_output"
        output_root = os.path.join("path/to/qwen2_output", output_dir_name, args.split)
    print(f"Output root: {output_root}")

    model, processor = load_model_and_processor()

    data_dir = os.path.join(args.data_root, args.split)
    visit_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for visit_id in tqdm(visit_ids, desc="visit_id"):
        print(f"\n{'='*60}\nvisit_id: {visit_id}\n{'='*60}")
        affordance_map = load_affordance_results(args.affordance_root, args.split, visit_id)
        visit_dir = os.path.join(data_dir, visit_id)
        video_ids = [d for d in os.listdir(visit_dir) if os.path.isdir(os.path.join(visit_dir, d))]
        print(f"Video(s): {len(video_ids)}")
        for video_id in video_ids:
            output_dir = os.path.join(output_root, visit_id)
            output_file = os.path.join(output_dir, f"{video_id}_point.json")
            if os.path.exists(output_file):
                print(f"  Skip {video_id} (already done)")
                continue
            print(f"  Processing {video_id}")
            try:
                results = process_single_video(
                    model, processor, args.data_root, args.split, visit_id, video_id,
                    args.clip_root, affordance_map, args.enable_validation
                )
                if results:
                    save_results(results, args.split, visit_id, video_id, output_root)
                else:
                    print(f"Skip {video_id}: no valid results")
            except Exception as e:
                print(f"Error processing {visit_id}/{video_id}: {e}")
                continue

if __name__ == "__main__":
    main() 