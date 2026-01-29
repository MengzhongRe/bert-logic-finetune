# scripts/data_processor.py
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from pathlib import Path

# 读取根目录config.yaml配置
from scripts.utils import cfg

class SentimentDataset(Dataset):
    """自定义数据集类，适配Hugging Face Trainer——张量默认留在CPU，Trainer自动移到GPU"""
    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels = labels
        # 移除手动传入的device参数，无需提前指定设备

    def __getitem__(self, idx: int) -> dict:
        """按索引取数据，张量留在CPU，Trainer自动处理设备分配"""
        item = {k: v[idx] for k, v in self.encodings.items()}  # 移除.to(self.device)
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)  # 移除.to(self.device)
        return item

    def __len__(self) -> int:
        return len(self.labels)

def load_single_data(file_path: str) -> tuple[list, list]:
    """加载单份CSV数据，提取text/label并做基础清洗"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    # 基础清洗：转字符串+去首尾空格，避免隐性脏数据
    texts = df["text"].astype(str).str.strip().tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels

def encode_texts(texts: list, tokenizer: AutoTokenizer) -> dict:
    """将原始文本编码为模型可识别的PyTorch张量，读取配置中max_seq_len"""
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=cfg["model"]["max_seq_len"],
        return_tensors="pt"
    )

def build_datasets(tokenizer: AutoTokenizer, device: torch.device) -> tuple[SentimentDataset, SentimentDataset, SentimentDataset]:
    """一站式构建训练/验证/测试数据集，供主程序调用——device仅传参不使用，保持接口兼容"""
    # 加载原始数据
    train_texts, train_labels = load_single_data(cfg["data"]["train_path"])
    val_texts, val_labels = load_single_data(cfg["data"]["val_path"])
    test_texts, test_labels = load_single_data(cfg["data"]["test_path"])
    # 打印数据统计
    print(f"📁 数据加载完成 | 训练集：{len(train_texts)}条 | 验证集：{len(val_texts)}条 | 测试集：{len(test_texts)}条")
    # 文本编码（张量默认在CPU）
    train_encodings = encode_texts(train_texts, tokenizer)
    val_encodings = encode_texts(val_texts, tokenizer)
    test_encodings = encode_texts(test_texts, tokenizer)
    # 构建数据集——不再传入device参数
    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)
    return train_dataset, val_dataset, test_dataset