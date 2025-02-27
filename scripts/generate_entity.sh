#!/bin/bash

MODEL_NAME="qwen2-7b-instruct"
LANGUAGE="en"

# 创建日志目录
mkdir -p ../log

# hf 本地模型运行
CUDA_VISIBLE_DEVICES=0 \
nohup python ../src/build_graph.py \
--generation_type "entity" \
--model_import_type "hf" \
--model_type "qwen" \
--hf_model_name "../model/llm/${MODEL_NAME}" \
--test_data_path "../data/benchmark/PatentMatch_${LANGUAGE}_parser.jsonl" \
--output_path "../data/benchmark/PatentMatch_${LANGUAGE}_output_${MODEL_NAME}.jsonl" \
--language "${LANGUAGE}" \
> ../log/generation_entity_${MODEL_NAME}.out  2>&1 &

echo "进程已在后台启动，进程ID: $!"

## api 模型运行
#nohup python ../src/build_graph.py \
#--generation_type "entity" \
#--model_import_type "api" \
#--api_model_name "${MODEL_NAME}" \
#--api_key "你的API密钥" \
#--test_data_path "../data/benchmark/PatentMatch_${LANGUAGE}_parser.jsonl" \
#--output_path "../data/benchmark/PatentMatch_${LANGUAGE}_output_${MODEL_NAME}.jsonl" \
#--language "${LANGUAGE}" \
#> ../log/generation_entity_${MODEL_NAME}.out  2>&1 &

#echo "进程已在后台启动，进程ID: $!"