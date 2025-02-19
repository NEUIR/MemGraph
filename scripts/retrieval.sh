#model_name="qwen2.5-7b-instruct"
#language="en"
#
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
#nohup python bge_retrieval.py \
#--instruction_template "Represent this sentence for searching relevant passages: [专利摘要]\n[扩展]" \
#--test_data_retrieval_path "./data/benchmark/PatentMatch_${language}_retrieval.json" \
#--keywords_data_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/qwen2.5-7b-instruct-10.2.json" \
#--corpus_path "./data/corpus/patent_${language}.json" \
#--language $language \
#--retrieval_output_path "/data1/xiongqiushi/main_code/patent_retrieval/data/retrieval_result/qwen2.5-7b-instruct-10.2.json" \
#--batch_size 1024 \
#--retrieval_model_path "/data1/xiongqiushi/model/bge/bge-base-${language}" \
#> ./log/bge_retrieval_${model_name}-10.2.out  2>&1 &

#--instruction_template "Represent this sentence for searching relevant passages: [专利摘要]\nTechnical Entities: \n[扩展]" \

#!/bin/bash

language="en"

export CUDA_VISIBLE_DEVICES=2,3,4

nohup python new_retrieval.py \
--instruction_template "Represent this sentence for searching relevant passages: [专利摘要]\nTechnical Entities: \n[扩展]" \
--test_data_retrieval_path "./data/benchmark/PatentMatch_${language}_retrieval.json" \
--keywords_data_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/glm-4-9b-chat-10.5-all.json" \
--corpus_path "./data/corpus/patent_${language}.json" \
--language $language \
--retrieval_output_path "/data1/xiongqiushi/main_code/patent_retrieval/data/retrieval_result/glm-4-9b-chat-10.5-all.json" \
--batch_size 512 \
--retrieval_model_path "/data1/xiongqiushi/model/bge/bge-base-${language}" \
> ./log/glm-4-9b-chat-10.5-all.out 2>&1 &

