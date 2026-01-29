# scripts/utils.py
import yaml
from pathlib import Path
import torch

def load_config(config_path: str = None) -> dict:
    """
    公共配置读取方法，默认读取根目录config.yaml
    :param config_path: 自定义配置文件路径，None则用默认
    :return: 配置字典
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# 全局配置实例，所有脚本直接导入即可
cfg = load_config()

def auto_detect_device() -> torch.device:
    """自动检测设备（GPU/CPU），读取配置中device参数"""
    if cfg["device"]["auto_detect"] and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device(cfg["device"]["manual_device"])