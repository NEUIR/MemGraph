

#llama
#CUDA_VISIBLE_DEVICES=3 \
#nohup python new_gen_keywords.py \
#--test_data_retrieval_path "/data1/xiongqiushi/main_code/patent_retrieval/data/benchmark/PatentMatch_zh_all.json" \
#--test_data_result_path "/data1/xiongqiushi/main_code/patent_retrieval/data/benchmark/PatentMatch_zh_all.json" \
#--keywords_output_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/Meta-Llama-3.1-8B-Instruct-entity-zh.json" \
#--language "zh" \
#--prompt_gen_keywords_zh_path "./prompt/gen_entity_zh.txt" \
#--prompt_gen_keywords_en_path "./prompt/gen_entity_en.txt" \
#--hf_model_name "/data3/xiongqiushi/model/Meta-Llama-3.1-8B-Instruct" \
#--model_type "llama" \
#> ./log/gen_keywords_Meta-Llama-3.1-8B-Instruct_zh.out  2>&1 &

#glm
#CUDA_VISIBLE_DEVICES=4 \
#nohup python new_gen_keywords.py \
#--test_data_retrieval_path "/data1/xiongqiushi/main_code/patent_retrieval/data/benchmark/PatentMatch_zh_all.json" \
#--test_data_result_path "/data1/xiongqiushi/main_code/patent_retrieval/data/benchmark/PatentMatch_zh_all.json" \
#--keywords_output_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/glm-4-9b-chat-entity-zh.json" \
#--prompt_gen_keywords_zh_path "./prompt/gen_entity_zh.txt" \
#--prompt_gen_keywords_en_path "./prompt/gen_entity_en.txt" \
#--language "zh" \
#--hf_model_name "/data3/xiongqiushi/model/glm-4-9b-chat" \
#--model_type "glm4" \
#> ./log/gen_keywords_glm-4-9b-chat_zh.out  2>&1 &

##internlm
#CUDA_VISIBLE_DEVICES=6 \
#nohup python new_gen_keywords.py \
#--test_data_retrieval_path "/data1/xiongqiushi/main_code/patent_retrieval/data/benchmark/PatentMatch_en_all.json" \
#--keywords_output_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/internlm2_5-7b-chat-10.5-all.json" \
#--test_data_result_path "/data1/xiongqiushi/main_code/patent_retrieval/data/benchmark/PatentMatch_en_all.json" \
#--language "en" \
#--prompt_gen_keywords_en_path "./prompt/10.5.txt" \
#--hf_model_name "/data3/xiongqiushi/model/internlm2_5-7b-chat" \
#--model_type "internlm" \
#> ./log/gen_keywords_internlm2_5-7b-chat_en_10.5.out  2>&1 &


#qwen
#CUDA_VISIBLE_DEVICES=6 \
#nohup python gen_keywords.py \
#--test_data_retrieval_path "./data/benchmark/PatentMatch_zh_retrieval.json" \
#--keywords_output_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/qwen2.5-7b-instruct-9.28-2.json" \
#--test_data_result_path "/data1/xiongqiushi/main_code/patent_retrieval/result/zh/glm-4-9b-chat/class_easy_glm-4-9b-chat-rag-1psg.json" \
#--prompt_gen_keywords_en_path "./prompt/9.28-2.txt" \
#--language "en" \
#--hf_model_name "/data3/xiongqiushi/model/Qwen2.5-7B-Instruct" \
#--model_type "qwen" \
#> ./log/gen_keywords_qwen2-7b-instruct_en.out  2>&1 &


#qwen
CUDA_VISIBLE_DEVICES=2 \
nohup python new_gen_keywords.py \
--model_import_type "api" \
--test_data_retrieval_path "./data1/xiongqiushi/main_code/patent_retrieval/data/benchmark/PatentMatch_en_all.json" \
--keywords_output_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/qwen25-14b-instruct-entity-en.json" \
--test_data_result_path "/data1/xiongqiushi/main_code/patent_retrieval/data/benchmark/PatentMatch_en_all.json" \
--prompt_gen_keywords_zh_path "./prompt/gen_entity_zh.txt" \
--prompt_gen_keywords_en_path "./prompt/gen_entity_en.txt" \
--language "en" \
--api_model_name "qwen2.5-14b-instruct" \
> ./log/qwen25_api_en-.out  2>&1 &

#yi
#CUDA_VISIBLE_DEVICES=7 \
#nohup python new_gen_keywords.py \
#--test_data_retrieval_path "/data1/xiongqiushi/main_code/patent_retrieval/data/benchmark/PatentMatch_en_all.json" \
#--keywords_output_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/yi-1.5-9b-chat-10.5-all.json" \
#--language "en" \
#--test_data_result_path "/data1/xiongqiushi/main_code/patent_retrieval/data/benchmark/PatentMatch_en_all.json" \
#--prompt_gen_keywords_en_path "./prompt/10.5.txt" \
#--hf_model_name "/data3/xiongqiushi/model/yi-1.5-9b-chat" \
#--model_type "yi" \
#> ./log/gen_keywords_yi-1.5-9b-chat_en-10.5.out  2>&1 &

