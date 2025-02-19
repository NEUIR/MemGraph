import sys
import os
import re
import json
from tqdm import tqdm
from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.utils import update_instruction, load_data, extract_abcd
from arguments import get_args
from .utils.call_openai import call_with_messages
from .utils.model_inference import model_load, model_inference


def extract_patent_abstract(text, language="zh"):
    """
    Extract patent abstract from text based on language-specific patterns.

    Args:
        text (str): Input text to process
        language (str): Language code ('zh' or 'en')

    Returns:
        str: Extracted patent abstract
    """
    if language == "en":
        pattern = r"Patent Abstract:\n(.*?)\n\nPlease extract"
    else:
        pattern = r"\n\n专利摘要：(.*?)\n\n请从这个专利摘要"

    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else "No patent abstract found in the given text."


def process_data(args, model=None, tokenizer=None, client=None):
    """
    Process patent data based on specified arguments and model type.

    Args:
        args: Command line arguments
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        client: OpenAI client for VLLM (optional)

    Returns:
        list: Processed results
    """
    # Load test data based on processing type
    if hasattr(args, 'test_data_result_path') and hasattr(args, 'test_data_retrieval_path'):
        test_data = load_data(args.test_data_result_path)
        retrieval_data = load_data(args.test_data_retrieval_path)
        process_type = "keywords"
    else:
        test_data = load_data(args.test_data_path)
        process_type = "classification"

    results = []
    n = 0

    for data in tqdm(test_data, desc="Processing", unit="sample"):
        # Determine prompt path based on language
        prompt_path = (
            f"args.prompt_gen_{process_type}_zh_path"
            if args.language == "zh"
            else f"args.prompt_gen_{process_type}_en_path"
        )

        with open(eval(prompt_path), 'r', encoding='utf-8') as prompt_file:
            prompt = prompt_file.read()

        # Format instruction based on process type
        if process_type == "keywords":
            new_instruction = prompt.format(abs=retrieval_data[n]["input"])
            n += 1
        else:
            # Extract all necessary abstracts and entities for classification
            abs = extract_patent_abstract(data["original_abstract"]["input_query"], args.language)
            entity = data["original_abstract"]["gen"]
            abstracts_entities = {}
            for letter in ['A', 'B', 'C', 'D']:
                abs_key = f"abs_{letter.lower()}"
                entity_key = f"entity_{letter.lower()}"
                abstracts_entities[abs_key] = extract_patent_abstract(
                    data[letter]["input_query"],
                    args.language
                )
                abstracts_entities[entity_key] = data[letter]["gen"]

            new_instruction = prompt.format(abs=abs, entity=entity, **abstracts_entities)

        # Generate response based on model type
        gen = None
        if args.model_import_type == "hf":
            gen = model_inference(args.model_type, new_instruction, model, tokenizer)
        elif args.model_import_type == "api":
            gen = call_with_messages(args.api_model_name, args.api_key, new_instruction, 5, 2)
        elif args.model_import_type == "vllm":
            chat_response = client.chat.completions.create(
                model="vllm-agent",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": new_instruction},
                ]
            )
            gen = chat_response.choices[0].message.content

        if gen is None:
            print("生成失败" if args.language == "zh" else "Generation failed")
            gen = "null"

        results.append({
            "input_query": abs if process_type == "classification" else new_instruction,
            "gen_classification": gen
        })

    return results


def main():
    args = get_args()

    # Initialize model based on import type
    model = tokenizer = client = None
    if args.model_import_type == "hf":
        model, tokenizer = model_load(args.hf_model_name, args.model_type)
    elif args.model_import_type == "vllm":
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{args.vllm_port}/v1",
        )

    # Process data
    results = process_data(args, model, tokenizer, client)

    # Determine output path
    output_path = (
        args.keywords_output_path
        if hasattr(args, 'keywords_output_path')
        else args.classification_output_path
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()