#!/bin/bash

# glm-4-9b-chat
# qwen2-7b-instruct
# Meta-Llama-3.1-8B-Instruct
# Qwen2.5-14B-Instruct

run_task() {
    local psg_num=$1
    local language="en"
    local model_name="qwen2-7b-instruct"
    local output_file="/data1/xiongqiushi/main_code/patent_retrieval/result/${language}/${model_name}/${model_name}-mg-${psg_num}psg-onlyzgen-true.json"
    local log_file="./log/model_inference_${psg_num}.out"
    local classification_output_path="/data1/xiongqiushi/main_code/patent_retrieval/gen_classification/en/qwen2-7b-instruct_10.5.json"
    local test_data_path="./data/benchmark/PatentMatch_${language}.jsonl"
#    local retrieval_results_path="/data1/xiongqiushi/main_code/patent_retrieval/data/keywords_output/zh/qwen25-14b-en.json"
    local retrieval_results_path="/data1/xiongqiushi/main_code/patent_retrieval/data/retrieval_result/only_bge/bge_top20_en.json"

    echo "Starting job for psg_num $psg_num"

    CUDA_VISIBLE_DEVICES=6 \
    nohup python inference-ppl.py \
    --psg_num "$psg_num" \
    --model_import_type "hf" \
    --api_model_name "qwen2.5-14b-instruct" \
    --output_path "$output_file" \
    --language "$language" \
    --hf_model_name "/data3/xiongqiushi/model/${model_name}" \
    --model_type "qwen" \
    --retrieval_results_path "$retrieval_results_path" \
    --test_data_path "$test_data_path"\
    --classification_output_path "$classification_output_path" \
    --use_classification 1 \
    > "$log_file" 2>&1 &

    local pid=$!
    echo "Job for psg_num $psg_num started with PID $pid"

    # 等待任务完成和输出文件生成
    while kill -0 $pid 2>/dev/null || [ ! -f "$output_file" ]; do
        sleep 30
        if ! kill -0 $pid 2>/dev/null; then
            echo "Process for psg_num $psg_num (PID $pid) has finished."
            if [ ! -f "$output_file" ]; then
                echo "Waiting for output file to be generated for psg_num $psg_num..."
            fi
        fi
    done

    echo "Job completed and output file generated for psg_num $psg_num."
}

#for psg_num in $(seq 0 1)
for psg_num in 3
do
    run_task $psg_num
done

echo "All jobs completed"