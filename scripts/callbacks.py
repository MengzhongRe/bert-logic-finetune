# scripts/callbacks.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import TrainerCallback
from scripts.utils import cfg
import matplotlib.font_manager as fm

class DirectPlotCallback(TrainerCallback):
    def __init__(self):
        self.log_dir = Path(cfg["model"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.log_dir / "live_training_plot.png"
        
        # 数据容器
        self.train_steps = []
        self.train_losses = []
        self.val_steps = []
        self.val_losses = []
        self.val_f1s = []
        
        # --- 字体设置优化 ---
        # 优先尝试加载系统自带的中文字体，如果找不到则回退到英文标签，防止乱码
        # 你可以在这里指定你本地具体的字体路径，例如: font_path = '/usr/share/fonts/SimHei.ttf'
        self.use_chinese = False 
        
        # 设置绘图风格
        sns.set_style("whitegrid")
        # 尝试设置通用无衬线字体
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return

        # 收集数据
        if "loss" in logs and "eval_loss" not in logs:
            self.train_steps.append(state.global_step)
            self.train_losses.append(logs["loss"])
            
        if "eval_loss" in logs:
            self.val_steps.append(state.global_step)
            self.val_losses.append(logs["eval_loss"])
            self.val_f1s.append(logs.get("eval_f1", 0))

        if len(self.train_steps) > 0:
            self._plot_and_save()

    def _plot_and_save(self):
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # --- 图1：Loss (使用英文标签) ---
            ax1.plot(self.train_steps, self.train_losses, label="Train Loss", color="#e74c3c", alpha=0.6)
            ax1.scatter(self.train_steps, self.train_losses, color="#e74c3c", s=10, alpha=0.6)

            if self.val_steps:
                ax1.plot(self.val_steps, self.val_losses, label="Val Loss", color="#3498db", linewidth=2)
                ax1.scatter(self.val_steps, self.val_losses, color="#3498db", s=30, marker="s", zorder=5)

            ax1.set_ylabel("Loss")
            ax1.set_title("Training & Validation Loss") # 英文标题
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # --- 图2：F1 Score (使用英文标签) ---
            if self.val_steps:
                ax2.plot(self.val_steps, self.val_f1s, label="Val F1 Score", color="#2ecc71", linewidth=2)
                ax2.scatter(self.val_steps, self.val_f1s, color="#2ecc71", s=30, marker="^", zorder=5)
                
                max_f1 = max(self.val_f1s)
                ax2.set_title(f"Validation F1 Score (Best: {max_f1:.4f})") # 英文标题
            else:
                ax2.text(0.5, 0.5, "Waiting for validation data...", ha='center')

            ax2.set_xlabel("Global Steps")
            ax2.set_ylabel("F1 Score")
            ax2.set_ylim(0, 1.05)
            ax2.legend()

            plt.tight_layout()
            plt.savefig(self.save_path, dpi=150)
            plt.close(fig)
            
        except Exception as e:
            print(f"Plotting Error: {e}")

# ... (保留之前的 import 和 DirectPlotCallback) ...
import logging
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

# 设置日志格式
logger = logging.getLogger(__name__)

class ConsolePrinterCallback(TrainerCallback):
    """
    自定义控制台打印回调 + 本地日志文件保存
    """
    def __init__(self):
        # 确定日志文件路径
        self.log_dir = Path(cfg["model"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training_console_output.txt"
        
        # 每次运行前清空旧日志（如果需要保留追加，去掉这行即可）
        # with open(self.log_file, "w", encoding="utf-8") as f:
        #     f.write("=== Training Log Start ===\n")

    def _log(self, message):
        """核心方法：同时打印到控制台和写入文件"""
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # 1. 训练日志 (Step, Loss, LR)
        if "loss" in logs and "eval_loss" not in logs:
            step = state.global_step
            total = state.max_steps
            loss = logs.get("loss", 0.0)
            lr = logs.get("learning_rate", 0.0)
            epoch = logs.get("epoch", 0.0)
            
            msg = f"📝 Step [{step}/{total}] | Epoch: {epoch:.2f} | Loss: {loss:.4f} | LR: {lr:.2e}"
            self._log(msg)

        # 2. 验证日志 (Table)
        if "eval_loss" in logs:
            epoch = logs.get("epoch", 0.0)
            eval_loss = logs.get("eval_loss", 0.0)
            acc = logs.get("eval_accuracy", 0.0)
            f1 = logs.get("eval_f1", 0.0)
            precision = logs.get("eval_precision", 0.0)
            recall = logs.get("eval_recall", 0.0)

            lines = [
                "",
                "="*60,
                f"📊 验证集评估报告 (Epoch {epoch:.2f})",
                "-" * 60,
                f"   📉 验证损失 (Loss)      : {eval_loss:.4f}",
                f"   🏆 F1 分数 (F1-Score)   : {f1:.4f}",
                f"   🎯 准确率 (Accuracy)    : {acc:.4f}",
                f"   🔍 精确率 (Precision)   : {precision:.4f}",
                f"   📡 召回率 (Recall)      : {recall:.4f}",
                "="*60,
                ""
            ]
            self._log("\n".join(lines))