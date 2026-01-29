# scripts/metrics.py
import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 读取根目录config.yaml配置
from scripts.utils import cfg

def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict:
    """
    计算二分类核心评估指标，适配Hugging Face Trainer
    后续扫参时，wandb会自动记录这些指标
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)  # 取logits最大值为预测标签
    # 计算核心指标，保留4位小数
    metrics = {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "f1": round(f1_score(labels, predictions, average="binary"), 4),
        "precision": round(precision_score(labels, predictions, average="binary"), 4),
        "recall": round(recall_score(labels, predictions, average="binary"), 4)
    }
    return metrics