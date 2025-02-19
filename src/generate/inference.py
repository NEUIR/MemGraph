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


args = get_args()

if args.model_import_type == "hf":
    model,tokenizer = model_load(args.hf_model_name, args.model_type)


if args.model_import_type == "vllm":
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.vllm_port}/v1",
    )
retrieval_results = load_data(args.retrieval_results_path)
test_data = load_data(args.test_data_path)

results = []
n = 0

for data in tqdm(test_data, desc="Processing", unit="sample"):
    instruction = data['instruction']
    retrieval_result = retrieval_results[n]
    if args.use_classification:
        prompt_path = "args.prompt_inference_" + f"{args.language}" + "_classification" + "_path"
        class_data = load_data(args.classification_output_path)[n]['gen_classification']
    else:
        prompt_path = "args.prompt_inference_" + f"{args.language}" + "_path"
        class_data=[]
    with open(eval(prompt_path), 'r', encoding='utf-8') as prompt:
        prompt = prompt.read()


    new_instruction = update_instruction(instruction, retrieval_result, args.psg_num, args.language,prompt,args.use_classification, class_data)
    n += 1
    #hf
    if args.model_import_type == "hf":
        outputs = model_inference(args.model_type,new_instruction,model,tokenizer)
        generated_answer = extract_abcd(outputs)
        if generated_answer is None:
            print("生成失败，使用默认答案")
            generated_answer = 'A'

    #api
    if args.model_import_type == "api":
        outputs = call_with_messages(args.api_model_name,args.api_key,new_instruction,5,2)
        if outputs is None:
            print("生成失败，使用默认答案")
            generated_answer = 'A'
        else:
            generated_answer = extract_abcd(outputs)
    #vllm
    if args.model_import_type == "vllm":
        chat_response = client.chat.completions.create(
            model="vllm-agent",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": new_instruction},
            ],
        )

        outputs = chat_response.choices[0].message.content
        generated_answer = extract_abcd(outputs)

        if generated_answer is None:
            chat_response = client.chat.completions.create(
                model="vllm-agent",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": new_instruction},
                ],
                seed=2024,
                temperature=0.1,
            )
            outputs = chat_response.choices[0].message.content
            generated_answer = extract_abcd(outputs)

            if generated_answer is None:
                print("生成失败，使用默认答案")
                generated_answer = 'A'
    is_same = generated_answer.strip() == data['output'].strip()


    results.append({
        "old_query": instruction,
        "new_query": new_instruction,
        "generated_answer": generated_answer,
        "expected_output": data['output'],
        "is_same": is_same
    })


# Ensure the target directory exists
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

# Save results to file
with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("Results saved to", args.output_path)