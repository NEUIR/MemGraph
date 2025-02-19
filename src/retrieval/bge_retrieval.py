from arguments import get_args
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
from utils.utils import load_data

def encode_texts(texts, model, tokenizer, device, batch_size):
    encoded_texts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**inputs)
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            encoded_batch = sentence_embeddings.cpu().numpy()
        encoded_texts.extend(encoded_batch)
    return np.array(encoded_texts)


def calculate_metrics(queries, query_encodings, corpus_encodings, corpus_texts, ground_truths, device,
                      top_k=[1,2,3,4,5, 10,20]):
    metrics = {f'{metric}@{k}': 0 for metric in ['recall', 'ndcg', 'precision'] for k in top_k}
    top_results = []

    for i, query_encoding in tqdm(enumerate(query_encodings), desc="Calculating metrics", total=len(query_encodings)):
        query_encoding_tensor = torch.from_numpy(query_encoding).float().to(device)
        corpus_encodings_tensor = torch.from_numpy(corpus_encodings).float().to(device)

        similarities = F.cosine_similarity(query_encoding_tensor.unsqueeze(0), corpus_encodings_tensor)
        top_indices = torch.topk(similarities, k=top_k[-1] + 1)
        top_indices_cpu = top_indices.indices.cpu().numpy()
        top_scores_cpu = top_indices.values.cpu().numpy()

        query_without_instruct = queries[i]

        # 简化的去重逻辑
        relevant_documents = []
        seen_scores = set()
        # for idx, score in zip(top_indices_cpu[1:], top_scores_cpu[1:]):
        for idx, score in zip(top_indices_cpu[1:], top_scores_cpu[1:]):  # 跳过第一个文档
            if score not in seen_scores:
                # relevant_documents.append({"passage": corpus_texts[idx][57:], "score": float(score)})
                relevant_documents.append({"passage": corpus_texts[idx], "score": float(score)})
                seen_scores.add(score)
            if len(relevant_documents) == top_k[-1]:
                break

        top_results.append({
            "query": query_without_instruct,
            "topk": relevant_documents
        })

        # 其余代码保持不变
        ground_truth = ground_truths[i]
        for k in top_k:
            retrieved_k = [doc["passage"] for doc in relevant_documents[:k]]
            if ground_truth in retrieved_k:
                metrics[f'recall@{k}'] += 1
                rank = retrieved_k.index(ground_truth)
                dcg = 1 / np.log2(rank + 2)
                idcg = 1
                metrics[f'ndcg@{k}'] += dcg / idcg
            metrics[f'precision@{k}'] += int(ground_truth in retrieved_k) / k

    for key in metrics:
        metrics[key] /= len(queries)

    return metrics, top_results
import re
def extract_query(test):
    text = test["gen_classification"]
    queries_match = re.search(r'增强查询描述：(.*?)$', text, re.DOTALL)
    result = queries_match.group(1).strip() if queries_match else ""
    if result == None:
        return test["input_query"]
    return result


def main():
    args = get_args()
    # Load data
    patent_datas = load_data(args.corpus_path)
    keywords_datas = load_data(args.keywords_data_path)
    test_datas = load_data(args.test_data_retrieval_path)

    # Initialize model
    tokenizer = AutoTokenizer.from_pretrained(args.retrieval_model_path)
    model = AutoModel.from_pretrained(args.retrieval_model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    new_query_datas = keywords_datas

    # For corpus
    abstract_texts = [patent["abstract_text"] for patent in patent_datas]
    abstract_encodings = encode_texts(abstract_texts, model, tokenizer, device, args.batch_size)

    instruction_template = args.instruction_template

    test_queries_ = [instruction_template.replace('[专利摘要]', test["input"])
                    for i, test in enumerate(test_datas)]
    test = [test for i, test in enumerate(new_query_datas)]
    test_queries = []
    for i,item in enumerate(test_queries_):
        test_queries.append(item.replace('[扩展]', test[i]['gen_classification']))
    test_outputs = [test["output"] for test in test_datas]
    test_encodings = encode_texts(test_queries_, model, tokenizer, device, args.batch_size)

    # Calculate metrics
    metrics, top_results = calculate_metrics(test_queries_, test_encodings, abstract_encodings, abstract_texts,
                                             test_outputs,device)

    # Save results
    output_data = pd.DataFrame(top_results)
    output_data.to_json(args.retrieval_output_path, orient='records', force_ascii=False, indent=4)

    # Print metrics
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
#
# from arguments import get_args
# import torch
# from transformers import AutoModel, AutoTokenizer
# from tqdm import tqdm
# import numpy as np
# import pandas as pd
# import torch.nn.functional as F
# from utils.utils import load_data
# from sklearn.metrics.pairwise import cosine_similarity
#
#
# def calculate_entropy(embedding, epsilon=1e-10):
#     sim_matrix = cosine_similarity(embedding.reshape(1, -1))
#     sim_matrix = np.clip(sim_matrix, epsilon, 1 - epsilon)  # 避免log(0)
#     entropy = -np.sum(sim_matrix * np.log2(sim_matrix))
#     return entropy
#
#
# def calculate_jensen_shannon_divergence(p, q, epsilon=1e-10):
#     p = np.clip(p, epsilon, 1 - epsilon)
#     q = np.clip(q, epsilon, 1 - epsilon)
#     m = 0.5 * (p + q)
#     return 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))
#
#
# def calculate_relevance_weighted_information_gain(query_emb, passage_emb, alpha=0.5, beta=10):
#     cosine_sim = cosine_similarity(query_emb.reshape(1, -1), passage_emb.reshape(1, -1))[0][0]
#
#     query_dist = cosine_similarity(query_emb.reshape(1, -1))
#     passage_dist = cosine_similarity(passage_emb.reshape(1, -1))
#
#     # 使用Jensen-Shannon散度代替KL散度
#     js_divergence = calculate_jensen_shannon_divergence(query_dist, passage_dist)
#
#     # JS散度的值域在0到1之间，不需要额外的缩放
#
#     # 结合余弦相似度和JS散度
#     weighted_gain = alpha * cosine_sim + (1 - alpha) * (1 - js_divergence)
#
#     return weighted_gain
#
#
# def calculate_metrics(queries, query_encodings, corpus_encodings, corpus_texts, ground_truths, device,
#                       top_k=[5, 10], relevance_threshold=0.5):
#     metrics = {f'{metric}@{k}': 0 for metric in ['recall', 'ndcg', 'precision'] for k in top_k}
#     top_results = []
#
#     for i, query_encoding in tqdm(enumerate(query_encodings), desc="Calculating metrics", total=len(query_encodings)):
#         query_encoding_tensor = torch.from_numpy(query_encoding).float().to(device)
#         corpus_encodings_tensor = torch.from_numpy(corpus_encodings).float().to(device)
#
#         similarities = F.cosine_similarity(query_encoding_tensor.unsqueeze(0), corpus_encodings_tensor)
#         top_indices = torch.topk(similarities, k=top_k[-1] + 1)
#         top_indices_cpu = top_indices.indices.cpu().numpy()
#         top_scores_cpu = top_indices.values.cpu().numpy()
#
#         query_without_instruct = queries[i]
#
#         relevant_documents = []
#         seen_scores = set()
#         for idx, score in zip(top_indices_cpu[1:], top_scores_cpu[1:]):  # 跳过第一个文档
#             if score not in seen_scores:
#                 passage_encoding = corpus_encodings[idx]
#                 relevance_weighted_gain = calculate_relevance_weighted_information_gain(query_encoding,
#                                                                                         passage_encoding)
#
#                 if relevance_weighted_gain > relevance_threshold:
#                     relevant_documents.append({
#                         "passage": corpus_texts[idx],
#                         "score": float(score),
#                         "relevance_weighted_gain": float(relevance_weighted_gain)
#                     })
#                     seen_scores.add(score)
#             if len(relevant_documents) == top_k[-1]:
#                 break
#
#         top_results.append({
#             "query": query_without_instruct,
#             "topk": relevant_documents
#         })
#
#         # 计算评估指标的代码保持不变
#         ground_truth = ground_truths[i]
#         for k in top_k:
#             retrieved_k = [doc["passage"] for doc in relevant_documents[:k]]
#             if ground_truth in retrieved_k:
#                 metrics[f'recall@{k}'] += 1
#                 rank = retrieved_k.index(ground_truth)
#                 dcg = 1 / np.log2(rank + 2)
#                 idcg = 1
#                 metrics[f'ndcg@{k}'] += dcg / idcg
#             metrics[f'precision@{k}'] += int(ground_truth in retrieved_k) / k
#
#     for key in metrics:
#         metrics[key] /= len(queries)
#
#     return metrics, top_results
#
#
# def encode_texts(texts, model, tokenizer, device, batch_size):
#     encoded_texts = []
#     for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
#         batch_texts = texts[i:i + batch_size]
#         inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
#         with torch.no_grad():
#             model_output = model(**inputs)
#             sentence_embeddings = model_output[0][:, 0]
#             sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
#             encoded_batch = sentence_embeddings.cpu().numpy()
#         encoded_texts.extend(encoded_batch)
#     return np.array(encoded_texts)
#
#
#
# def main():
#     args = get_args()
#     # Load data
#     patent_datas = load_data(args.corpus_path)
#     keywords_datas = load_data(args.keywords_data_path)
#     test_datas = load_data(args.test_data_retrieval_path)
#
#     # Initialize model
#     tokenizer = AutoTokenizer.from_pretrained(args.retrieval_model_path)
#     model = AutoModel.from_pretrained(args.retrieval_model_path)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if torch.cuda.device_count() > 1:
#         model = torch.nn.DataParallel(model)
#     model.to(device)
#     model.eval()
#
#     # Prepare data
#     abstract_texts = [patent["abstract_text"] for patent in patent_datas]
#     abstract_encodings = encode_texts(abstract_texts, model, tokenizer, device, args.batch_size)
#
#     instruction_template = args.instruction_template
#     test_queries = [instruction_template.replace('[专利摘要]', test["input"])
#                     for i, test in enumerate(test_datas)]
#     test_outputs = [test["output"] for test in test_datas]
#     test_encodings = encode_texts(test_queries, model, tokenizer, device, args.batch_size)
#
#     # Calculate metrics
#     metrics, top_results = calculate_metrics(test_queries, test_encodings, abstract_encodings, abstract_texts,
#                                              test_outputs, device, relevance_threshold=0.01)
#
#     # Save results
#     output_data = pd.DataFrame(top_results)
#     output_data.to_json(args.retrieval_output_path, orient='records', force_ascii=False, indent=4)
#
#     # Print metrics
#     print("Metrics:", metrics)
#
#
# if __name__ == "__main__":
#     main()