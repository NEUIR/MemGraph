import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import json
from utils import load_data, extract_abcd, update_instruction
from tqdm import tqdm
from arguments import get_args
from call_openai import call_with_messages
from model_inference import model_load, model_inference
from prompt import *


def main():
    # 获取命令行参数
    args = get_args()

    # 初始化模型（如果使用HF模型）
    if args.model_import_type == "hf":
        model, tokenizer = model_load(args.hf_model_name, args.model_type)
    else:
        model, tokenizer = None, None

    # 加载检索结果和测试数据
    retrieval_results = load_data(args.retrieval_output_path)
    test_data = load_data(args.test_data_path)

    # 选择语言对应的提示模板
    prompt = inference_zh if args.language == "zh" else inference_en

    results = []
    n = 0

    # 处理每个样本
    for data in tqdm(test_data, desc="Processing", unit="sample"):
        # 检索结果
        retrieval_result = retrieval_results[n]

        # 更新指令添加检索结果、实体和本体信息
        new_instruction = update_instruction(data["original_instruction"], retrieval_result, args.psg_num, args.language, prompt,
                                             data["ontology"])
        n += 1

        # 根据模型类型进行推理
        if args.model_import_type == "hf" and model and tokenizer:
            outputs = model_inference(args.model_type, new_instruction, model, tokenizer)
            generated_answer = extract_abcd(outputs)
            if generated_answer is None:
                print("生成失败，使用默认答案")
                generated_answer = 'A'

        elif args.model_import_type == "api":
            outputs = call_with_messages(args.api_model_name, args.api_key, new_instruction, 5, 2)
            if outputs is None:
                print("生成失败，使用默认答案")
                generated_answer = 'A'
            else:
                generated_answer = extract_abcd(outputs)
                if generated_answer is None:
                    generated_answer = 'A'

        # 检查答案是否正确
        is_same = generated_answer.strip() == data['output'].strip()

        # 收集结果
        results.append({
            "new_instruction": new_instruction,
            "generated_answer": generated_answer,
            "expected_output": data['output'],
            "is_same": is_same
        })

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 保存结果到文件
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 计算准确率
    correct_count = sum(1 for result in results if result['is_same'])

    print("Results saved to", args.output_path)
    print("Accuracy:", correct_count)


if __name__ == "__main__":
    main()