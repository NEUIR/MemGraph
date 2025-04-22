# MemGraph: Enhancing the Patent Matching Capability of Large Language Models via Memory Graph

Source code for our SIGIR'25 paper: 
[Enhancing the Patent Matching Capability of Large Language Models via Memory Graph](https://arxiv.org/abs/2504.14845)

If you find this work useful, please cite our paper and give us a shining star 🌟

## 🎯 Overview
We propose MemGraph, a method that augments the patent matching capabilities of LLMs by incorporating a memory graph derived from their parametric memory. 

Specifically, MemGraph prompts LLMs to traverse their memory to identify relevant entities within patents, followed by attributing these entities to corresponding ontologies. After traversing the memory graph, we utilize extracted entities and ontologies to improve the capability of LLM in comprehending the semantics of patents. 

Experimental results on the PatentMatch dataset demonstrate the effectiveness of MemGraph, achieving a **17.68%** performance improvement over baseline LLMs.

![model](https://newxqsoss.oss-cn-hangzhou.aliyuncs.com/undefinedmodel.png)

## ⚙️ Environment Setup
1️⃣ Clone from git:
```bash
git clone https://github.com/NEUIR/MemGraph.git
cd MemGraph
```
2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Preparation
1️⃣ Download retrieval corpus we collected from the [Google Drive](https://drive.google.com/drive/folders/1TBvQTIEDsUW6bKFKGSg9yM8wvio5wMIO?usp=sharing), create a `corpus` directory and make sure that the files under the data folder contain the following before running:
```
data/
├── corpus/
│   ├── patent_en.json
│   └── patent_zh.json 
└── benchmark/
```
2️⃣ Data Processing

```bash
sh scripts/data_processing.sh
```
3️⃣ Our implementation requires both embedding and language models. First, create a `models` directory in the project root and download the necessary models from Hugging Face:

```
models/
├── embedding/
│   ├── bge-base-en/       # English embedding model
│   └── bge-base-zh/       # Chinese embedding model
└── llm/
    └── Qwen2-7B-Instruct/ # Language model
```

You can download these models from:
- Embedding models: [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en), [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh)
- Language Model: [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) (or other compatible LLMs like [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B), [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [GLM-4-9B-Chat](https://huggingface.co/THUDM/glm-4-9b-chat))

## 🧑‍💻 Reproduce
1️⃣ Build MemGraph

```bash
# Generate entity
sh scripts/generate_entity.sh

# Generate ontology
sh scripts/generate_ontology.sh
```

2️⃣ Retrieval with MemGraph
```bash
sh scripts/retrieval.sh
```

3️⃣ Inference with MemGraph
```bash
sh scripts/inference.sh
```

## 📝 Citation

```bibtex
@inproceedings{xiong2025enhancing,
  title={Enhancing the Patent Matching Capability of Large Language Models via Memory Graph},
  author={Xiong, Qiushi and Xu, Zhipeng and Liu, Zhenghao and Wang, Mengjia and Chen, Zulong and Sun, Yue and Gu, Yu and Li, Xiaohua and Yu, Ge},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2025}
}
```

## 📨 Contact

If you have questions, suggestions, and bug reports, please email:
```
xiongqiushi@stumail.neu.edu.cn
```








