import os
import json
import torch
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

class QwenAffordanceModel:
    """
    Qwen模型用于Affordance推理任务
    """
    def __init__(self, model_path='Qwen/Qwen2.5-7B-Instruct'):
        """
        初始化Qwen模型
        Args:
            model_path (str): 模型路径
        """
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.AFFORDANCE_PROMPT = """You are an expert in human-computer interaction and affordance analysis. \
Your task is to identify the specific functional component (affordance) that a person would interact with to perform a given action, and output ONLY the name of the functional component.\n\nGiven a description of an action, provide ONLY the name of the functional component. Be concise and direct.\n\nExamples:\n- Action: \"Close the bedroom door\" → Affordance: handle\n- Action: \"Turn on the light\" → Affordance: switch\n- Action: \"Open the refrigerator\" → Affordance: handle\n- Action: \"Flush the toilet\" → Affordance: button\n- Action: \"Adjust the thermostat\" → Affordance: dial\n- Action: \"Lock the front door\" → Affordance: lock\n- Action: \"Open the window\" → Affordance: handle\n- Action: \"Start the washing machine\" → Affordance: button\n\nNow, analyze this action and identify the functional component:\n\nAction: \"{action_description}\"\n\nAffordance:"""

    def generate_response(self, prompt, max_new_tokens=256):
        """
        生成文本响应
        Args:
            prompt (str): 输入提示
            max_new_tokens (int): 最大生成token数
        Returns:
            str: 模型响应
        """
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True
        ).to('cuda')
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]

    def infer_affordance(self, action_description):
        """
        根据动作描述推理出功能部件(Affordance)
        Args:
            action_description (str): 动作描述
        Returns:
            dict: 包含推理结果的结果字典
        """
        try:
            prompt = self.AFFORDANCE_PROMPT.format(action_description=action_description)
            response = self.generate_response(prompt)
            affordance = response.strip()
            if "Affordance:" in affordance:
                affordance = affordance.split("Affordance:")[-1].strip()
            affordance = affordance.replace('"', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
            affordance = ' '.join(affordance.split())
            
            # 后处理：提取功能部件名称，去除"of object"部分
            if ' of ' in affordance:
                affordance = affordance.split(' of ')[0].strip()
            
            return {
                "description": action_description,
                "affordance": affordance
            }
        except Exception as e:
            print(f"Error in affordance inference: {e}")
            return {
                "description": action_description,
                "affordance": "",
                "error": str(e)
            }

    def process_description_file(self, desc_file_path, visit_id):
        """
        处理单个描述文件
        Args:
            desc_file_path (str): 描述文件路径
            visit_id (str): visit_id
        Returns:
            list: 处理结果列表
        """
        results = []
        try:
            with open(desc_file_path, 'r', encoding='utf-8') as f:
                desc_data = json.load(f)
            descriptions = desc_data.get('descriptions', [])
            print(f"Processing {len(descriptions)} descriptions for visit_id: {visit_id}")
            for desc in descriptions:
                desc_id = desc.get('desc_id', '')
                description = desc.get('description', '')
                if description:
                    result = self.infer_affordance(description)
                    result['desc_id'] = desc_id
                    results.append(result)
                    print(f"  desc_id: {desc_id}")
                    print(f"  description: {description}")
                    print(f"  affordance: {result['affordance']}")
                    print("-" * 50)
        except Exception as e:
            print(f"Error processing description file {desc_file_path}: {e}")
        return results

def process_affordance_inference(data_root, split, model_path='Qwen/Qwen2.5-7B-Instruct'):
    """
    处理affordance推理的主函数
    Args:
        data_root (str): 数据根目录路径
        split (str): 数据集分割类型 ('train', 'val', 'test')
        model_path (str): 模型路径
    """
    print(f"开始处理affordance推理...")
    print(f"数据根目录: {data_root}")
    print(f"数据集分割: {split}")
    print(f"模型路径: {model_path}")
    print("正在加载Qwen模型...")
    model = QwenAffordanceModel(model_path)
    print("模型加载完成")
    benchmark_dir = os.path.join(data_root, "benchmark_file_lists")
    split_file = os.path.join(benchmark_dir, f"{split}_set.csv")
    if not os.path.exists(split_file):
        print(f"错误: 找不到分割文件 {split_file}")
        return
    print(f"读取分割文件: {split_file}")
    df = pd.read_csv(split_file)
    print(f"找到 {len(df)} 个场景")
    df['visit_id'] = df['visit_id'].astype(str)
    df_single = df.drop_duplicates(subset=['visit_id'], keep='first')
    print(f"去重后剩余 {len(df_single)} 个唯一visit_id")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'affordance_result', split)
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    for index, row in tqdm(df_single.iterrows(), total=len(df_single), desc='处理visit_id进度'):
        visit_id = row['visit_id']
        desc_file_path = os.path.join(data_root, split, visit_id, f"{visit_id}_descriptions.json")
        if os.path.exists(desc_file_path):
            print(f"\n处理 visit_id: {visit_id}")
            results = model.process_description_file(desc_file_path, visit_id)
            output_file_path = os.path.join(output_dir, f"{visit_id}_affordance.json")
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {output_file_path}")
            all_results.extend(results)
        else:
            print(f"警告: 描述文件不存在 {desc_file_path}")
    print(f"\n处理完成！")
    print(f"总共处理了 {len(all_results)} 个描述")
    print(f"结果文件保存在: {output_dir}")

def main():
    """
    主函数，只支持affordance推理模式
    """
    parser = argparse.ArgumentParser(description='使用Qwen模型进行Affordance推理')
    parser.add_argument('--mode', type=str, choices=['affordance'], required=True,
                       help='运行模式: 仅支持affordance(推理功能部件)')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                       help='Qwen模型路径 (默认: Qwen/Qwen2.5-7B-Instruct)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据根目录路径 (affordance模式需要)')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True,
                       help='数据集分割类型 (affordance模式需要)')
    args = parser.parse_args()
    if args.mode == 'affordance':
        process_affordance_inference(args.data_root, args.split, args.model_path)

#推理模式：python qwen/qwen.py --mode affordance --data_root data/raw_data --split val
if __name__ == '__main__':
    main()
