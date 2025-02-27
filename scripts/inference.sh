#!/bin/bash

MODEL_NAME="qwen2-7b-instruct"
LANGUAGE="en"

# 创建所需目录
mkdir -p ../output

# hf 本地模型运行
CUDA_VISIBLE_DEVICES=0 \
nohup python ../src/inference.py \
--model_import_type "hf" \
--model_type "qwen" \
--hf_model_name "../model/llm/${MODEL_NAME}" \
--retrieval_output_path "../data/retrieval_result/${MODEL_NAME}.json" \
--test_data_path "../data/benchmark/PatentMatch_${LANGUAGE}_output_${MODEL_NAME}.jsonl" \
--psg_num 3 \
--language "${LANGUAGE}" \
--output_path "../output/output_${LANGUAGE}_${MODEL_NAME}.json" \
> ../log/inference_${MODEL_NAME}.out  2>&1 &

echo "进程已在后台启动，进程ID: $!"

## api 模型运行
#nohup python ../src/inference.py \
#--model_import_type "api" \
#--api_model_name "${MODEL_NAME}" \
#--api_key "你的API密钥" \
#--retrieval_output_path "../data/retrieval_result/${MODEL_NAME}" \
#--test_data_path "../data/benchmark/PatentMatch_${LANGUAGE}_output_${MODEL_NAME}.jsonl" \
#--psg_num 3 \
#--language "${LANGUAGE}" \
#--output_path "../output/output_${LANGUAGE}_${MODEL_NAME}.json" \
#> ../log/inference_${MODEL_NAME}.out  2>&1 &
#
#echo "进程已在后台启动，进程ID: $!"