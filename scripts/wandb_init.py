# scripts/wandb_init.py
import wandb
import os
import torch  # 导入 torch 以检测实际设备
from scripts.utils import cfg

def init_wandb(project_name: str = "bert_logic_finetune", run_name: str = None):
    """
    初始化WandB，同步训练超参数、指标到云端
    """
    if run_name is None:
        run_name = cfg["train"].get("run_name", "roberta_logic_finetune")
    
    # 初始化 WandB
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=cfg,
        save_code=True,
        resume="never"
    )
    
    # ✅ 修复核心：使用 torch 检测真实设备，并改名为 "runtime_device"
    # 避免覆盖 config.yaml 中 "device" 字典导致的类型冲突报错
    real_device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        real_device = "mps"

    wandb.config.update({
        "train_samples": 1518,
        "val_samples": 434,
        "test_samples": 217,
        "runtime_device": real_device  # <--- 关键修改：换个键名，不要叫 "device"
    }, allow_val_change=True) # 加上这个参数作为双重保险
    
    return run

def finish_wandb():
    wandb.finish()