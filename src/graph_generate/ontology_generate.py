import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import re
import json
from ..utils.utils import load_data
from tqdm import tqdm
from ..utils.arguments import get_args
from ..utils.call_openai import call_with_messages
from openai import OpenAI
from ..utils.model_inference import model_load, model_inference


def extract_patent_abstract(text):
    if args.language == "en":
        pattern = r"Patent Abstract:\n(.*?)\n\nPlease extract"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return "No patent abstract found in the given text."
    else:
        pattern = r"\n\n专利摘要：(.*?)\n\n请从这个专利摘要"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return "No patent abstract found in the given text."

args = get_args()

if args.model_import_type == "hf":
    model,tokenizer = model_load(args.hf_model_name, args.model_type)
if args.model_import_type == "vllm":
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.vllm_port}/v1",
    )

test_data = load_data(args.test_data_path)

results = []
n = 0

for data in tqdm(test_data, desc="Processing", unit="sample"):
    if args.language == "zh":
        prompt_path = "args.prompt_gen_classification_zh_path"
    else:
        prompt_path = "args.prompt_gen_classification_en_path"
    with open(eval(prompt_path), 'r', encoding='utf-8') as prompt:
        prompt = prompt.read()
    abs = extract_patent_abstract(data["original_abstract"]["input_query"])
    entity = data["original_abstract"]["gen"]
    abs_a = extract_patent_abstract(data["A"]["input_query"])
    entity_a = data["A"]["gen"]
    abs_b = extract_patent_abstract(data["B"]["input_query"])
    entity_b = data["B"]["gen"]
    abs_c = extract_patent_abstract(data["C"]["input_query"])
    entity_c = data["C"]["gen"]
    abs_d = extract_patent_abstract(data["D"]["input_query"])
    entity_d = data["D"]["gen"]
    new_instruction = prompt.format(abs=abs,entity=entity,abs_a=abs_a,entity_a=entity_a,abs_b=abs_b,entity_b=entity_b,abs_c=abs_c,entity_c=entity_c,abs_d=abs_d,entity_d=entity_d)
    #hf
    if args.model_import_type == "hf":
        gen = model_inference(args.model_type,new_instruction,model,tokenizer)
        if gen is None:
            print("生成失败")
            gen = "null"

    #api
    if args.model_import_type == "api":
        gen = call_with_messages(args.api_model_name,args.api_key,new_instruction,5,2)
        if gen is None:
            print("生成失败，使用默认答案")

    if args.model_import_type == "vllm":
        chat_response = client.chat.completions.create(
            model="vllm-agent",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": new_instruction},
            ],
        )

        gen = chat_response.choices[0].message.content


        if gen is None:
            print("生成失败")

    results.append({
        "input_query": abs,
        "gen_classification": gen
    })


# Ensure the target directory exists
os.makedirs(os.path.dirname(args.classification_output_path), exist_ok=True)

# Save results to file
with open(args.classification_output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("Results saved to", args.classification_output_path)