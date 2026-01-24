# 📊 Logical Sentiment Analysis: Probing BERT's Reasoning Boundaries
# 基于逻辑语义的情感分析研究

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow?style=flat-square)](https://huggingface.co/)
[![Analysis](https://img.shields.io/badge/Analysis-CheckList-blue?style=flat-square)]()

> **Project Goal:** To evaluate whether Deep Learning models (BERT/RoBERTa) truly "understand" logic, or if they merely rely on shallow heuristics (keyword matching).

## 📖 Introduction (项目简介)

在传统的 NLP 任务中，模型往往能取得极高的 Accuracy。然而，作为逻辑学背景的研究者，我发现模型在面对**“双重否定”**、**“反讽”**等复杂逻辑句式时，表现往往不尽如人意。

本项目基于 **Data-Centric AI** 的理念，利用 LLM (DeepSeek) 生成高质量的对抗性测试样本，对中文情感分析模型进行了**“行为测试” (Behavioral Testing)**。

## 🛠️ Methodology (方法论)

1.  **Data Generation (数据构建):** 
    - 使用 `DeepSeek-V3.2` API，通过 Prompt Engineering 生成了 **400** 条包含特定逻辑结构的测试数据。
    - 涵盖类型：`Simple` (简单句), `Double Negation` (双重否定), `Concessive` (转折), `Irony` (反讽)。
2.  **Inference (推理):** 
    - 加载 `uer/roberta-base-finetuned-dianping-chinese` 作为基座模型。
    - 编写 Python 脚本进行批量推理与 Logits 分析。
3.  **Evaluation (评测):** 
    - 计算各句式类别的 Accuracy。
    - 收集各句式的`Bad Cases`分析误判原因。

## 📊 Experiment Results (实验结果)

**Base Model:** `uer/roberta-base-finetuned-dianping-chinese`
**Test Set Size:** 400 samples

| Logic Type (句式类型) | Accuracy (准确率) | Analysis (归因分析) |
| :--- | :--- | :--- |
| **Simple (简单句)** | **98.18%** | ✅ **Baseline.** 模型基础能力扎实。 |
| **Concessive (转折)** | **98.99%** | ✅ **Attention Success.** 自注意力机制成功捕捉了句子的重心（如“但是”之后的内容）。 |
| **Double Negation (双重否定)** | **77.42%** | ⚠️ **Logic Failure.** 模型在较复杂的句式中难以处理 $\neg(\neg P) = P$ 的逻辑抵消，易受否定词干扰。 |
| **Irony (反讽)** | **57.00%** | ❌ **Semantic Collapse.** 准确率接近随机猜测。模型过度依赖浅层关键词（如“聪明”、“棒”），无法理解语言中的言外之意。 |

## 🧠 Key Insights (核心洞察)

1.  **Shallow Heuristics (浅层特征依赖):** BERT 类模型在判断情感时，更多是基于“词袋模型”式的关键词权重叠加，而非真正的语义理解。
2.  **The Limits of Encoders:** 对于反讽这种需要**世界知识 (World Knowledge)** 和 **反事实推理** 的任务，仅靠 Encoder 架构的分类头是远远不够的。
3.  **Future Work:** 引入 **Chain-of-Thought (CoT)** 数据进行微调，或者结合 **RAG** 引入外部常识库，可能是解决反讽识别的有效路径。

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data (Optional):**
   ```bash
   python scripts/gen_data_deepseek.py
   ```

3. **Run Evaluation:**
   ```bash
   python test_v3.py
   ```

---
*Author: Yi Meng (SYSU Logic Master Student)*