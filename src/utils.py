import os
import json
import re


def extract_abcd(text):
    """
    从文本中提取 ABCD 选项答案
    支持以下格式:
    - 单个字母 'A'/'B'/'C'/'D'
    - '答案：A' 或 'Answer: A'
    - '[A]' 或其他包含 ABCD 的形式

    Args:
        text (str): 输入文本
    Returns:
        str: 提取到的答案(A/B/C/D)，未找到则返回 None
    """
    if not text:
        return None

    # 1. 先检查最后一个非空字符
    text = text.strip()
    if text and text[-1].upper() in 'ABCD':
        return text[-1].upper()

    # 2. 如果最后一个字符不是答案，使用正则匹配
    patterns = [
        r'(?:答案|Answer)\s*[:：]?\s*(?:\[)?([A-D])(?:\])?',  # 匹配"答案：A"或"Answer: A"
        r'\[([A-D])\]',  # 匹配 [A]
        r'([A-D])'  # 匹配单个字母
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def update_instruction(instruction,retrieval_data,prepend_retrieval_results_num,language,prompt, class_data):
    i = 0
    passage_list=[]
    if language == 'zh':
        for passage in retrieval_data['topk'][:prepend_retrieval_results_num]:
            i += 1
            passage = f"专利{i}:"+ passage['passage']
            passage_list.append(passage)
        top_n_passages = "\n".join(passage_list)
        new_instruction = prompt.format(num=prepend_retrieval_results_num, rag_passages=top_n_passages,question=instruction,classification=class_data)
    else :
        for passage in retrieval_data['topk'][:prepend_retrieval_results_num]:
            i += 1
            passage = f"Patent{i}:"+ passage['passage']
            passage_list.append(passage)
        top_n_passages = "\n".join(passage_list)
        new_instruction = prompt.format(num=prepend_retrieval_results_num, rag_passages=top_n_passages,
                                        question=instruction,classification=class_data)

    return new_instruction


def load_data(path):
    data_list = []
    # 获取文件扩展名
    _, file_extension = os.path.splitext(path)
    with open(path, 'r', encoding='utf-8') as json_file:
        if file_extension.lower() == '.json':
            # 如果是 JSON 文件
            data = json.load(json_file)
            data_list = data
        elif file_extension.lower() == '.jsonl':
            # 如果是 JSON Lines 文件
            for line in json_file:
                data = json.loads(line)
                data_list.append(data)
        else:
            # 未知文件类型
            raise ValueError(f"Unsupported file extension: {file_extension}")

    return data_list

