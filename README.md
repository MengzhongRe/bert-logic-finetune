
# 🧠 BERT-Logic-Finetune: 专注中文复杂逻辑的情感分析增强模型

<div align="center">

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/YiMeng-SYSU/roberta-logic-sentiment-zh)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/YiMeng-SYSU/chinese-logic-sentiment-dataset)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

**通过 "LLM生成-数据清洗-增量微调" 闭环，解决传统 BERT 模型在反讽、双重否定等复杂中文语境下的"智商掉线"问题。**

[快速开始](#-快速开始) • [核心效果](#-核心效果) • [方法论](#-项目方法论) • [项目结构](#-项目结构) • [复现指南](#-复现指南)

</div>

---

## 📖 项目简介 (Introduction)

在中文情感分析任务中，传统的预训练模型（如 BERT/RoBERTa）往往在面对**隐晦表达**时表现不佳。例如：
- **反讽 (Irony)**：“这部剧简直有毒，害得我昨晚又熬到凌晨三点，黑眼圈都出来了！”（模型常误判为负面，实为正面）
- **双重否定 (Double Negative)**：“不得不说，这次体验真的没法让人不失望。”（逻辑嵌套导致模型混乱）

本项目基于 `uer/roberta-base-finetuned-dianping-chinese`，构建了一套**基于 Doubao API 的高质量逻辑增强数据集**，针对性地修复了模型在**反讽（正话反说/反话正说）、双重否定、转折**等高难度句式上的缺陷。

## 🚀 核心效果 (Performance)

在独立构建的 **960条“金标准”测试集 (Golden Test Set)** 上（该测试集经过严格人工清洗，且未参与训练），模型取得了显著的性能飞跃。

### 1. 总体指标对比
| 指标 (Metric) | 微调前 (Baseline) | **微调后 (Ours)** | 提升幅度 |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 81.98% | **96.15%** | 🔺 **+14.17%** |
| **Macro-F1** | 77.96% | **96.00%** | 🔺 **+18.04%** |
| **Recall (Negative)** | 77.86% | **97.53%** | 🔺 **+19.67%** |
| **Recall (Positive)** | 82.00% | **94.15%** | 🔺 **+12.15%** |

### 2. 分句式逻辑能力提升
> **这是本项目最大的亮点**：彻底解决了模型“听不懂赖话”、“看不懂反话”的问题。

| 句式类型 (Logic Type) | 微调前 Acc | **微调后 Acc** | 关键突破 |
| :--- | :--- | :--- | :--- |
| **反讽 (Irony)** | 63.60% | **92.47%** | 正面反讽召回率从 **25%** 飙升至 **65%+**，模型学会了“反话正说”。 |
| **双重否定 (Double Neg)** | 82.08% | **96.67%** | 完美区分“不得不赞”与“不得不骂”。 |
| **转折 (Transition)** | 87.55% | **97.93%** | 不再被前半句的干扰信息带偏。 |
| **简单句 (Simple)** | 94.58% | **97.50%** | 基础能力未退化，反而更稳固。 |

### 3. 领域泛化能力
特别针对 **出行旅游** 和 **生活服务** 领域的“阴阳怪气”进行了补强训练，出行旅游领域准确率从 **71%** 提升至 **97.5%**。

---

## 🛠 快速开始 (Quick Start)

### 使用 Hugging Face 直接推理

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 1. 加载模型
model_name = "YiMeng-SYSU/roberta-logic-sentiment-zh"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2. 准备测试文本 (正面反讽示例)
text = "这部剧简直有毒，害得我昨晚又熬到凌晨三点，黑眼圈都出来了!"

# 3. 推理
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

# 4. 解析结果
id2label = {0: "Negative (负面)", 1: "Positive (正面)"}
pred_id = torch.argmax(probs).item()
confidence = probs.max().item()

print(f"文本: {text}")
print(f"预测: {id2label[pred_id]} (置信度: {confidence:.4f})")
# 输出: Positive (正面)
```

---

## 🔬 项目方法论 (Methodology)

本项目采用 **"Diagnostic-Driven Data Augmentation" (诊断驱动的数据增强)** 策略：

1.  **初始生成 (Batch 1-3)**：利用 Doubao API 生成基础数据，涵盖6大领域、4种句式。
2.  **构建金标准 (Golden Set)**：人工清洗 Batch 1 (960条)，作为**永不参与训练**的测试集，确立评估基准。
3.  **诊断分析 (Diagnosis)**：在 Baseline 上评估，发现 **"正面反讽 (Recall 25%)"** 和 **"出行旅游领域"** 是最大短板。
4.  **特攻增强 (Hard Boost)**：设计特殊的 Prompt (CoT + Few-Shot)，专门生成 **800条** 高难度样本（如"抱怨式夸奖"、"消极双重否定"）。
5.  **增量微调 (Finetuning)**：混合基础集与增强集，使用极低学习率 (`3e-5`) 进行微调，避免灾难性遗忘。

---

## 📂 项目结构 (Project Structure)

```text
bert-logic-finetune/
├── config.yaml             # 项目全局配置
├── main.py                 # 模型训练主入口
├── manual_eval.py          # 模型评估脚本 (加载模型 -> 跑测试集 -> 出报告)
├── requirements.txt        # 依赖包列表
├── data_gen/               # 【数据工程】核心代码
│   ├── data_generator_doubao.py    # 调用 LLM API 生成数据
│   └── merge_data.py       # 数据熔炼与清洗
├── scripts/                # 【工具库】
│   ├── evaluator.py        # 详细的指标计算逻辑 (Macro-F1, Recall0/1)
│   └── utils.py            # 通用工具
|   └── .....           
└── results/                # 【评估产物】
    ├── before_finetune/    # 微调前的评估报告 & Bad Cases
    └── after_finetune/     # 微调后的评估报告 & Bad Cases
```

---

## 💻 复现指南 (Reproduction)

### 1. 环境准备
```bash
git clone https://github.com/MengzhongRe/bert-logic-finetune.git
cd bert-logic-finetune
pip install -r requirements.txt
```

### 2. 数据生成 (可选)
如果你想复现数据生成过程（需要自备 API Key）：
```bash
export ARK_API_KEY="your_api_key"
python data_gen/data_generator_doubao.py
python data_gen/gen_aug_data.py
```

### 3. 模型训练
```bash
# 默认使用 data/train/train.csv 进行训练
python main.py
```

### 4. 模型评估
```bash
# 评估训练好的模型在 Golden Test Set 上的表现
python manual_eval.py
```

---

## 🔗 资源链接

- **🤗 Model Weights**: [YiMeng-SYSU/roberta-logic-sentiment-zh](https://huggingface.co/YiMeng-SYSU/roberta-logic-sentiment-zh)
- **📚 Dataset**: [YiMeng-SYSU/chinese-logic-sentiment-dataset](https://huggingface.co/datasets/YiMeng-SYSU/chinese-logic-sentiment-dataset)

## 🤝 致谢
- Base Model: [uer/roberta-base-finetuned-dianping-chinese](https://huggingface.co/uer/roberta-base-finetuned-dianping-chinese)
- Data Generation: [Doubao API (Volcengine)](https://www.volcengine.com/product/doubao)

---
*Created by [MengzhongRe](https://github.com/MengzhongRe)*
