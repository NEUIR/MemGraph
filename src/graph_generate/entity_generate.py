import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import json
from ..utils.utils import load_data
from tqdm import tqdm
from ..utils.arguments import get_args
from ..utils.call_openai import call_with_messages
from ..utils.model_inference import model_load, model_inference


args = get_args()

if args.model_import_type == "hf":
    model,tokenizer = model_load(args.hf_model_name, args.model_type)

test_data = load_data(args.test_data_result_path)

results = []
n = 0

for data in tqdm(test_data, desc="Processing", unit="sample"):
    if args.language == "zh":
        prompt_path = "args.prompt_gen_keywords_zh_path"
    else:
        prompt_path = "args.prompt_gen_keywords_en_path"
    with open(eval(prompt_path), 'r', encoding='utf-8') as prompt:
        prompt = prompt.read()
    new_instruction = prompt.format(abs=test_data[n]["input"])
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