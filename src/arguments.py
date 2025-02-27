import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # 语言
    parser.add_argument("--language", type=str, default='en', help="Language code (en or zh)")

    # 生成类型参数
    parser.add_argument("--generation_type", type=str, default=None,
                        choices=["entity", "ontology"],
                        help="Type of generation to perform (entity or ontology)")

    # 检索相关参数
    parser.add_argument("--corpus_path", type=str,
                        default="./data/corpus/patent_en.json",
                        help="Path to patent data JSON file")
    parser.add_argument("--test_data_path", type=str,
                        default="./data/benchmark/PatentMatch_en.jsonl",
                        help="Path to preprocessed test data")
    parser.add_argument("--retrieval_model_path", type=str,
                        default="./models/embedding/bge-base-en",
                        help="Path to the embedding model")
    parser.add_argument("--retrieval_output_path", type=str,
                        default="./output/retrieval_results.json",
                        help="Path to save retrieval results")
    parser.add_argument("--batch_size", type=int,
                        default=32,
                        help="Batch size for retrieval")

    # 模型导入参数
    parser.add_argument("--model_import_type", type=str, default="hf",
                        choices=["hf", "vllm", "api"],
                        help="Model import type: hf (Hugging Face), vllm, or api")
    parser.add_argument("--hf_model_name", type=str,
                        default="./models/llm/Qwen2-7B-Instruct",
                        help="Path to the local HF model")
    parser.add_argument("--model_type", type=str, default="qwen",
                        choices=["qwen", "qwen25", "glm", "llama"],
                        help="Type of model architecture")
    parser.add_argument("--api_model_name", type=str,
                        default="qwen2-7b-instruct",
                        help="API model name")
    parser.add_argument("--api_key", type=str,
                        help="API key for hosted models")

    # 推理参数
    parser.add_argument("--output_path", type=str,
                        default='./output/inference_results.json',
                        help="Path to save inference results")
    parser.add_argument("--psg_num", type=int, default=3,
                        help="Number of passages to use")

    return parser.parse_args()