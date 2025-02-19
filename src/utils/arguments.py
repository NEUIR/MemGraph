import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # prompt
    parser.add_argument("--prompt_inference_zh_path", type=str, default="/data1/xiongqiushi/main_code/patent_retrieval/prompt/rag_zh.txt", help="prompt_inference_zh_path")
    parser.add_argument("--prompt_inference_zh_classification_path", type=str, default="/data1/xiongqiushi/main_code/patent_retrieval/prompt/rag_zh_classification.txt", help="prompt_inference_zh_classfication_path")
    parser.add_argument("--prompt_inference_en_path", type=str, default="./prompt/rag_en.txt", help="prompt_inference_en_path")
    parser.add_argument("--prompt_inference_en_classification_path", type=str, default="./prompt/rag_en_classification.txt", help="prompt_inference_en_classification_path")
    parser.add_argument("--prompt_gen_classification_zh_path", type=str, default="./prompt/try.txt", help="gen_classification_zh_one")
    parser.add_argument("--prompt_gen_classification_en_path", type=str, default="./prompt/gen_classification_en_one.txt", help="gen_classification_en_one")
    parser.add_argument("--prompt_gen_keywords_zh_path", type=str,
                        default="./prompt/keywords_gen_zh_short.txt", help="gen_classification_zh_one")
    parser.add_argument("--prompt_gen_keywords_en_path", type=str,
                        default="./prompt/9.26.txt", help="gen_classification_en_one")
    parser.add_argument("--prompt_try", type=str,
                        default="./prompt/try.txt", help="gen_classification_en_one")

    # retrieval arguments
    parser.add_argument("--corpus_path", type=str,
                        default="./data/corpus/patent_en.json",
                        help="Path to patent data JSON file")
    parser.add_argument("--keywords_data_path", type=str,
                        default="/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/qwen25-14b-instruct-entity-en.json",
                        help="Path to keywords data JSON file")
    parser.add_argument("--test_data_retrieval_path", type=str,
                        default="./data/benchmark/PatentMatch_en_retrieval.json",
                        help="Path to test data JSON file")
    parser.add_argument("--test_data_result_path", type=str,
                        default="/data1/xiongqiushi/main_code/patent_retrieval/data/benchmark/PatentMatch_en_all.json",
                        help="Path to test data JSON file")
    parser.add_argument("--retrieval_model_path", type=str,
                        default="/data3/xiongqiushi/model/bge/bge-base-en",
                        help="Path to the pre-trained model")
    # parser.add_argument("--instruction_template", type=str,
    #                     default="为这个句子生成表示以用于检索相关文章：[专利摘要]\n技术实体：[扩展]",
    #                     help="Instruction template")
    # parser.add_argument("--instruction_template", type=str,
    #                     default="为这个文章生成表示以用于检索相关文章：[专利摘要]",
    #                     help="Instruction template")
    parser.add_argument("--instruction_template", type=str,
                        default="Represent this sentence for searching relevant passages: [专利摘要]\nTechnical Entities: \n[扩展]",
                        help="Instruction template")
    parser.add_argument("--retrieval_output_path", type=str,
                        default="/data1/xiongqiushi/main_code/patent_retrieval/data/keywords_output/zh/qwen25-14b-en.json",
                        help="Path to save the output JSON file")
    parser.add_argument("--batch_size", type=int,
                        default=32,
                        help="batch size of retrieval")

    # model import arguments
    parser.add_argument("--model_import_type", type=str, default="hf", help="hf or vllm or api")
    parser.add_argument("--hf_model_name", type=str, default="/data3/xiongqiushi/model/yi-1.5-9b-chat",
                        help="Path to the model")
    parser.add_argument("--model_type", type=str, default="yi",
                        help="qwen/glm4/gemma/yi/internlm/baichuan/llama/phi")
    parser.add_argument("--vllm_port", type=str, default="2222", help="vllm port")
    parser.add_argument("--api_model_name", type=str, default="qwen2-7b-instruct", help="api model name")
    parser.add_argument("--api_key", type=str, default="sk-c134394cdf084a7eb5cd42e164ad1962", help="api_key")

    # inference arguments
    parser.add_argument("--retrieval_results_path", type=str,
                        default="/data1/xiongqiushi/main_code/patent_retrieval/data/keywords_output/zh/llama/bge_top10_result_zh+fankui.json",
                        help="Path to retrieval results")
    parser.add_argument("--test_data_path", type=str,
                        default="/data1/xiongqiushi/main_code/patent_retrieval/data/benchmark/PatentMatch_en.jsonl",help="Path to save output")
    parser.add_argument("--output_path", type=str,
                        default='/data1/xiongqiushi/main_code/patent_retrieval/result/zh/Meta-Llama-3.1-8B-Instruct/yi-1.5-9b-chat-1psg-k.json',
                        help="Path to save output")
    parser.add_argument("--classification_output_path", type=str,
                        default='/data1/xiongqiushi/main_code/patent_retrieval/gen_classification/zh/yi-1.5-9b-chat_easy.json',
                        help="Path to save output")
    parser.add_argument("--keywords_output_path", type=str,
                        default='/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/yi-1.5-9b-chat-9.15.json',
                        help="Path to save output")
    parser.add_argument("--new_output_path", type=str,
                        default='/data1/xiongqiushi/main_code/patent_retrieval/gen_new/zh/glm-4-9b-chat-9.8.json',
                        help="Path to save output")
    parser.add_argument("--psg_num", type=int, default=0, help="Number of passages")
    parser.add_argument("--language", type=str, default='en', help="Language code")
    parser.add_argument("--use_classification", type=int, default=0, help="use_classification")

    return parser.parse_args()