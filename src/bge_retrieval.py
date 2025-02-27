from arguments import get_args
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
import json
import os
import torch.nn.functional as F
from utils import load_data


def encode_texts(texts, model, tokenizer, device, batch_size):
    """将文本编码为向量表示"""
    encoded_texts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="编码文本"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**inputs)
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            encoded_batch = sentence_embeddings.cpu().numpy()
        encoded_texts.extend(encoded_batch)
    return np.array(encoded_texts)


def perform_retrieval(queries, query_encodings, corpus_encodings, corpus_texts, device, top_k=5):
    """执行检索并返回每个查询的前k个结果"""
    retrieval_results = []

    for i, query_encoding in tqdm(enumerate(query_encodings), desc="执行检索", total=len(query_encodings)):
        query_encoding_tensor = torch.from_numpy(query_encoding).float().to(device)
        corpus_encodings_tensor = torch.from_numpy(corpus_encodings).float().to(device)

        # 计算余弦相似度
        similarities = F.cosine_similarity(query_encoding_tensor.unsqueeze(0), corpus_encodings_tensor)
        top_indices = torch.topk(similarities, k=top_k + 1)  # +1 是为了跳过可能的自身
        top_indices_cpu = top_indices.indices.cpu().numpy()
        top_scores_cpu = top_indices.values.cpu().numpy()

        # 创建检索结果
        relevant_documents = []
        seen_documents = set()  # 用于去重

        # 跳过第一个结果（可能是查询自身）并获取前k个文档
        for idx, score in zip(top_indices_cpu[1:], top_scores_cpu[1:]):
            doc_text = corpus_texts[idx]
            # 使用文本内容作为去重键
            if doc_text not in seen_documents:
                relevant_documents.append({
                    "passage": doc_text,
                    "score": float(score)
                })
                seen_documents.add(doc_text)

            if len(relevant_documents) >= top_k:
                break

        retrieval_results.append({
            "query": queries[i],
            "topk": relevant_documents
        })

    return retrieval_results


def main():
    args = get_args()

    # 加载专利语料库
    patent_corpus = load_data(args.corpus_path)

    # 加载预处理过的测试数据（包含实体和本体信息）
    processed_data = load_data(args.test_data_path)

    # 初始化模型
    tokenizer = AutoTokenizer.from_pretrained(args.retrieval_model_path)
    model = AutoModel.from_pretrained(args.retrieval_model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # 为语料库编码
    abstract_texts = [patent["abstract_text"] for patent in patent_corpus]
    abstract_encodings = encode_texts(abstract_texts, model, tokenizer, device, args.batch_size)

    # 根据语言选择指令模板
    if args.language == "zh":
        instruction_template = "为这个句子生成表示以用于检索相关文章：[专利摘要]\n技术实体：[扩展]"
    else:
        instruction_template = "Represent this sentence for searching relevant passages: [专利摘要]\nTechnical Entities: \n[扩展]"

    # 构建查询
    test_queries = []
    for item in processed_data:
        # 使用原始专利和其实体进行查询扩展
        query = instruction_template.replace('[专利摘要]', item['original_patent'])
        query = query.replace('[扩展]', item['entity']['original'])
        test_queries.append(query)

    # 编码查询
    query_encodings = encode_texts(test_queries, model, tokenizer, device, args.batch_size)

    # 执行检索
    retrieval_results = perform_retrieval(
        test_queries,
        query_encodings,
        abstract_encodings,
        abstract_texts,
        device,
        top_k=args.top_k if hasattr(args, 'top_k') else 5
    )

    # 将检索结果添加到原始数据中
    for i, item in enumerate(processed_data):
        item['retrieval_results'] = retrieval_results[i]['topk']

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.retrieval_output_path), exist_ok=True)

    # 保存结果
    with open(args.retrieval_output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"检索完成，结果已保存至: {args.retrieval_output_path}")


if __name__ == "__main__":
    main()