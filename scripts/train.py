# scripts/train.py
import os
import torch
import pandas as pd
from pathlib import Path
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from torch.utils.data import Dataset
import torch.nn.functional as F
import wandb
from transformers.trainer_callback import PrinterCallback

from scripts.metrics import compute_metrics
from scripts.wandb_init import init_wandb, finish_wandb
from scripts.utils import cfg
from scripts.callbacks import DirectPlotCallback, ConsolePrinterCallback
from scripts.evaluator import run_detailed_eval

# 1. 统一使用 Config 中的路径
# 之前是: log_save_dir = Path("./train_logs")
# ✅ 改为:
output_dir = Path(cfg["model"]["output_dir"])
log_dir = Path(cfg["model"]["log_dir"])

os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

class CustomTrainer(Trainer):
    # ... (CustomTrainer 代码完全不变，此处省略) ...
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = None
        if isinstance(inputs, dict) and "labels" in inputs:
            labels = inputs.pop("labels")
        elif "labels" in kwargs:
            labels = kwargs.get("labels")
        if isinstance(inputs, dict):
            outputs = model(**inputs)
        else:
            outputs = model(*inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(input=logits.view(-1, 2), target=labels.view(-1), reduction="mean")
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else torch.tensor(0.0, device=logits.device)
        return (loss, outputs) if return_outputs else loss

def init_trainer(
    model: PreTrainedModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: torch.device
) -> Trainer:
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    batch_size = int(train_cfg["batch_size"])
    learning_rate = float(train_cfg["learning_rate"])
    num_epochs = int(train_cfg["num_epochs"])
    weight_decay = float(train_cfg["weight_decay"])
    logging_steps = int(train_cfg["logging_steps"])
    early_patience = int(train_cfg["early_stopping_patience"])
    save_limit = int(train_cfg["save_total_limit"])
    fp16_enabled = train_cfg["fp16"] and device.type == "cuda"
    run_name = train_cfg.get("run_name", "roberta_logic_finetune")

    training_args = TrainingArguments(
        output_dir=model_cfg["output_dir"],
        logging_dir=model_cfg["log_dir"],
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_limit,
        load_best_model_at_end=True,
        metric_for_best_model=train_cfg["main_metric"],
        greater_is_better=True,
        fp16=fp16_enabled,
        report_to="wandb",
        disable_tqdm=True,  # 关闭进度条
        run_name=run_name,
        logging_strategy="steps",
        remove_unused_columns=False
    )

    init_wandb(project_name="bert_logic_finetune", run_name=training_args.run_name)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_patience),
            DirectPlotCallback(),
            ConsolePrinterCallback(),
        ]
    )
    trainer.remove_callback(PrinterCallback)
    return trainer

def run_training(
    trainer: Trainer,
    test_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase
) -> None:
    model_name = cfg["model"]["model_name"]
    # 2. 这里也要用统一的 output_dir
    # output_dir = cfg["model"]["output_dir"] (已经在开头定义了 Path 对象)
    
    print(f"\n🚀 开始微调 {model_name} 模型...")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"✅ 训练完成！最优模型已保存至：{output_dir}")

    print(f"\n📊 测试集评估最优模型效果...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    
    print(f"🔍 测试集核心指标（情感二分类）：")
    for metric, value in test_metrics.items():
        if metric.startswith("eval_"):
            print(f"   {metric.replace('eval_', '').capitalize()}: {value:.4f}")

    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

    tokenizer.save_pretrained(output_dir)
    print(f"\n🎉 分词器已保存至：{output_dir}")

    # 3. CSV 保存路径修正
    # 使用统一的 log_dir (即 roberta_training_logs)
    log_history = trainer.state.log_history
    train_logs_df = pd.DataFrame([log for log in log_history if "loss" in log and "eval_loss" not in log])
    val_logs_df = pd.DataFrame([log for log in log_history if "eval_loss" in log])
    test_logs_df = pd.DataFrame([test_metrics])

    if not train_logs_df.empty:
        train_logs_df.to_csv(log_dir / "train_logs.csv", index=False, encoding="utf-8-sig")
    if not val_logs_df.empty:
        val_logs_df.to_csv(log_dir / "val_logs.csv", index=False, encoding="utf-8-sig")
    if not test_logs_df.empty:
        test_logs_df.to_csv(log_dir / "test_logs.csv", index=False, encoding="utf-8-sig")
    
    # 修改提示信息
    print(f"\n📝 训练/验证/测试 CSV日志已保存至：{log_dir}")
    # ✅ 新增：在一切结束后，运行详细的“体检”
    # ==========================================
    # 获取 trainer 当前使用的 device (cuda/mps/cpu)
    device = trainer.args.device 
    
    # 传入训练好的模型进行深度评估
    run_detailed_eval(trainer.model, tokenizer, device)

    finish_wandb()