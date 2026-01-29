# scripts/model.py
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

# 读取根目录config.yaml配置
from scripts.utils import cfg

def load_model_and_tokenizer(device: torch.device) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """加载预训练模型和分词器，读取config.yaml配置"""
    model_name = cfg["model"]["model_name"]
    num_labels = cfg["model"]["num_labels"]
    # 加载分词器（自动匹配基座模型规则）
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 加载序列分类模型（自动新增二分类输出层）
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=num_labels,
        problem_type="text_classification"  # 明确任务类型，避免损失函数推断错误
    ).to(device)
    # 打印模型/设备/批次信息（批次从train配置读取）
    print(f"⚙️  模型加载完成 | 预训练模型：{model_name}")
    print(f"⚙️  训练设备：{device} | 批次大小：{cfg['train']['batch_size']}")
    return model, tokenizer