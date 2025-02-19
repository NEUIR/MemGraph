##glm
#CUDA_VISIBLE_DEVICES=0 \
#nohup python ontology_generate.py \
#--prompt_gen_classification_zh_path "./prompt/gen_ontology_zh.txt" \
#--prompt_gen_classification_en_path "./prompt/gen_ontology_en.txt" \
#--test_data_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/glm-4-9b-chat-entity-zh.json" \
#--classification_output_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_classification/en/glm-4-9b-chat_zh.json" \
#--language "zh" \
#--hf_model_name "/data3/xiongqiushi/model/glm-4-9b-chat" \
#--model_type "glm4" \
#> ./log/gen_classification_glm-4-9b-chat_zh.out  2>&1 &

#internlm
#CUDA_VISIBLE_DEVICES=5 \
#nohup python ontology_generate.py \
#--prompt_gen_classification_en_path "./prompt/fenlei_en_10.4.txt" \
#--test_data_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/internlm2_5-7b-chat-10.5-all.json" \
#--classification_output_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_classification/en/internlm2_5-7b-chat_10.5.json" \
#--language "en" \
#--hf_model_name "/data3/xiongqiushi/model/internlm2_5-7b-chat" \
#--model_type "internlm" \
#> ./log/gen_classification_internlm2_5-7b-chat_10.5.out  2>&1 &

##yi
#CUDA_VISIBLE_DEVICES=6 \
#nohup python ontology_generate.py \
#--prompt_gen_classification_en_path "./prompt/fenlei_en_10.4.txt" \
#--test_data_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/yi-1.5-9b-chat-10.5-all.json" \
#--classification_output_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_classification/en/yi-1.5-9b-chat_10.5.json" \
#--language "en" \
#--hf_model_name "/data3/xiongqiushi/model/yi-1.5-9b-chat" \
#--model_type "yi" \
#> ./log/gen_classification_yi-1.5-9b-chat-10.5.out  2>&1 &

#llama
#CUDA_VISIBLE_DEVICES=6 \
#nohup python ontology_generate.py \
#--prompt_gen_classification_zh_path "./prompt/gen_ontology_zh.txt" \
#--prompt_gen_classification_en_path "./prompt/gen_ontology_en.txt" \
#--test_data_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/Meta-Llama-3.1-8B-Instruct-entity-zh.json" \
#--classification_output_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_classification/en/Meta-Llama-3.1-8B-Instruct_zh.json" \
#--language "zh" \
#--hf_model_name "/data3/xiongqiushi/model/Meta-Llama-3.1-8B-Instruct" \
#--model_type "llama" \
#> ./log/gen_classification_Meta-Llama-3-8B-Instruct_zh.out  2>&1 &

##qwen
CUDA_VISIBLE_DEVICES=2 \
nohup python gen_classification.py \
--model_import_type "api" \
--prompt_gen_classification_zh_path "./prompt/gen_ontology_zh.txt" \
--prompt_gen_classification_en_path "./prompt/gen_ontology_en.txt" \
--test_data_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_keywords/zh/qwen25-14b-instruct-entity-en.json" \
--classification_output_path "/data1/xiongqiushi/main_code/patent_retrieval/gen_classification/en/qwen25-14b-instruct_en.json" \
--language "en" \
> ./log/gen_classification_qwen25-14b-instruct_en.out  2>&1 &