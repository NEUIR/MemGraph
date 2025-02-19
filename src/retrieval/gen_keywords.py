import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.utils import update_instruction, load_data, extract_abcd
from tqdm import tqdm
import os
from arguments import get_args
from api.api import call_with_messages
from openai import OpenAI
import torch
from model_inference import model_load, model_inference

import re


def extract_content(text):
    # 定义正则表达式模式，匹配“参考以下专利摘要，回答后续问题：”和“问题如下：”之间的内容
    pattern = r"参考以下专利摘要，回答后续问题：\n(.*?)\n问题如下：\n"
    # 使用 re.findall 提取匹配的内容
    match = re.findall(pattern, text, re.DOTALL)

    if match:
        return match[0].strip()  # 返回匹配的内容，并去除多余的空白字符
    else:
        return None


args = get_args()

if args.model_import_type == "hf":
    model,tokenizer = model_load(args.hf_model_name, args.model_type)

test_data = load_data(args.test_data_result_path)

test_data_ = load_data(args.test_data_retrieval_path)

results = []
n = 0

for data in tqdm(test_data, desc="Processing", unit="sample"):
    if args.language == "zh":
        prompt_path = "args.prompt_gen_keywords_zh_path"
    else:
        prompt_path = "args.prompt_gen_keywords_en_path"
    with open(eval(prompt_path), 'r', encoding='utf-8') as prompt:
        prompt = prompt.read()
    new_instruction = prompt.format(abs=test_data_[n]["input"])
    # new_instruction = prompt.format(benchmark=da  ta['old_query'], abs=test_data_[n]["input"],
    #                                 result=extract_content(data["new_query"]))
    n+=1

    #hf
    if args.model_import_type == "hf":
        gen = model_inference(args.model_type,new_instruction,model,tokenizer)
        if gen is None:
            print("生成失败")
            gen = "null"

    #api
    if args.model_import_type == "api":
        outputs = call_with_messages(args.api_model_name,args.api_key,new_instruction,5,2)
        if outputs is None:
            print("生成失败，使用默认答案")

    results.append({
        "input_query": new_instruction,
        "gen_classification": gen
    })


# Ensure the target directory exists
os.makedirs(os.path.dirname(args.keywords_output_path), exist_ok=True)

# Save results to file
with open(args.keywords_output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("Results saved to", args.keywords_output_path)