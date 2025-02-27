import sys
import os
import json
from tqdm import tqdm

# 添加父目录到路径以导入相关模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from prompt import *
from utils import load_data
from arguments import get_args
from call_openai import call_with_messages
from model_inference import model_load, model_inference


def generate_text(args, instruction, model=None, tokenizer=None):
    """使用指定的模型生成文本。"""
    gen = None

    if args.model_import_type == "hf" and model and tokenizer:
        gen = model_inference(args.model_type, instruction, model, tokenizer)
        if gen is None:
            print("生成失败")
            gen = "null"

    elif args.model_import_type == "api":
        gen = call_with_messages(args.api_model_name, args.api_key, instruction, 5, 2)
        if gen is None:
            print("生成失败，使用默认答案")
            gen = "null"

    return gen


def process_entity_generation(args, model=None, tokenizer=None):
    """处理实体生成，直接修改输入文件，添加entity字段。"""
    try:
        # 读取预处理后的文件
        input_file_path = args.test_data_path
        data = load_data(input_file_path)

        # 选择合适的提示模板路径
        if args.language == "zh":
            prompt_template = gen_entity_zh
        else:
            prompt_template = gen_entity_en

        # 对每个题目进行处理
        for item in tqdm(data, desc="处理实体生成", unit="样本"):
            # 创建entity字段
            item['entity'] = {}

            # 为原始专利生成实体
            original_instruction = prompt_template.format(abs=item['original_patent'])
            original_entity = generate_text(args, original_instruction, model, tokenizer)
            item['entity']['original'] = original_entity

            # 为每个选项生成实体
            for option in ['A', 'B', 'C', 'D']:
                option_instruction = prompt_template.format(abs=item[f'option_{option}'])
                option_entity = generate_text(args, option_instruction, model, tokenizer)
                item['entity'][option] = option_entity

        # 保存回同一文件
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"实体生成完成，结果已保存至: {args.output_path}")

    except Exception as e:
        print(f"实体生成处理异常: {e}")
        sys.exit(1)


def process_ontology_generation(args, model=None, tokenizer=None):
    """处理本体生成，使用entity字段生成ontology字段。"""
    try:
        # 读取包含实体的文件
        input_file_path = args.test_data_path
        data = load_data(input_file_path)

        # 选择合适的提示模板路径
        if args.language == "zh":
            prompt_template = gen_ontology_zh
        else:
            prompt_template = gen_ontology_en

        # 检查是否已经生成了实体
        if not data or 'entity' not in data[0]:
            print("错误: 请先运行实体生成步骤，确保数据中包含entity字段")
            sys.exit(1)

        # 对每个题目进行处理
        for item in tqdm(data, desc="处理本体生成", unit="样本"):
            # 构建指令
            instruction = prompt_template.format(
                abs=item['original_patent'],
                entity=item['entity']['original'],
                abs_a=item['option_A'],
                entity_a=item['entity']['A'],
                abs_b=item['option_B'],
                entity_b=item['entity']['B'],
                abs_c=item['option_C'],
                entity_c=item['entity']['C'],
                abs_d=item['option_D'],
                entity_d=item['entity']['D']
            )

            # 生成本体
            ontology = generate_text(args, instruction, model, tokenizer)

            # 添加本体字段
            item['ontology'] = ontology

        # 保存回同一文件
        with open(input_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"本体生成完成，结果已保存至原文件: {input_file_path}")

    except Exception as e:
        print(f"本体生成处理异常: {e}")
        sys.exit(1)


def main():
    # 获取命令行参数
    args = get_args()

    # 检查生成类型参数
    if not hasattr(args, 'generation_type'):
        print("错误: 请使用 --generation_type 参数指定生成类型 (entity 或 ontology)")
        sys.exit(1)

    # 初始化模型
    model, tokenizer = None, None

    if args.model_import_type == "hf":
        try:
            model, tokenizer = model_load(args.hf_model_name, args.model_type)
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)

    # 根据生成类型执行不同的处理
    if args.generation_type == "entity":
        process_entity_generation(args, model, tokenizer)
    elif args.generation_type == "ontology":
        process_ontology_generation(args, model, tokenizer)
    else:
        print(f"错误: 不支持的生成类型: {args.generation_type}，请使用 --generation_type entity 或 ontology")
        sys.exit(1)


if __name__ == "__main__":
    main()