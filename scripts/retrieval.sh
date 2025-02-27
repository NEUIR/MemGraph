#!/bin/bash

# 设置参数
LANGUAGE="en"
MODEL_NAME="qwen2-7b-instruct"

# 创建所需目录
mkdir -p ../data/retrieval_result

echo "开始执行检索任务: ${MODEL_NAME}..."

CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup python ../src/bge_retrieval.py \
  --retrieval_model_path "../model/embedding/bge-base-${LANGUAGE}" \
  --test_data_path "../data/benchmark/PatentMatch_${LANGUAGE}_output_${MODEL_NAME}.jsonl" \
  --corpus_path "../data/corpus/patent_${LANGUAGE}.json" \
  --language $LANGUAGE \
  --retrieval_output_path "../data/retrieval_result/${MODEL_NAME}.json" \
  --batch_size 512 \
> ./log/retrieval_${MODEL_NAME}.out 2>&1 &

echo "任务已在后台启动，日志文件: ./log/retrieval_${MODEL_NAME}.out"

