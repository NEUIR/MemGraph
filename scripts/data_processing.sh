#!/bin/bash

# 解析英文专利数据
python ../src/data_processing.py ../data/benchmark/PatentMatch_en.jsonl ../data/benchmark/PatentMatch_en_parser.jsonl

# 解析中文专利数据
python ../src/data_processing.py ../data/benchmark/PatentMatch_zh.jsonl ../data/benchmark/PatentMatch_zh_parser.jsonl