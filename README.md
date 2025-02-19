# Enhancing the Patent Matching Capability of Large Language Models via Memory Graph

Source code for our paper: 
[Enhancing the Patent Matching Capability of Large Language Models via Memory Graph](https://arxiv.org)

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

## 📚 Data Preparation
1️⃣ Download retrieval corpus we collected from the [Google Drive](https://drive.google.com/drive/folders/1TBvQTIEDsUW6bKFKGSg9yM8wvio5wMIO?usp=sharing), please make sure that the files under the data folder contain the following before running:

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
@inproceedings{,
  title={Enhancing the Patent Matching Capability of Large Language Models via Memory Graph},
  author={},
  booktitle={},
  pages={},
  year={2025}
}
```

## 📨 Contact

If you have questions, suggestions, and bug reports, please email:
```
xiongqiushi@stumail.neu.edu.cn
```








