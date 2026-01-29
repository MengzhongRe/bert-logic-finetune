# scripts/evaluator.py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scripts.utils import cfg

test_path = cfg["data"]["test_path"]
# 基准线（用于对比提升幅度）
BASELINE = {
    "logic": {"简单句": 0.9250, "双重否定": 0.8750, "转折关系": 0.8583, "反讽": 0.8208},
    "domain": {"影视娱乐": 0.9375, "美食餐饮": 0.9000, "购物消费": 0.8750, "生活服务": 0.8688, "出行旅游": 0.8438, "日常社交": 0.7938},
    "hard_cases": {"反讽_日常社交": 0.6750, "双重否定_生活服务": 0.7500}
}

def print_diff(new_acc, old_acc):
    """生成带涨跌符号的字符串"""
    diff = new_acc - old_acc
    if diff > 0: return f"{new_acc:.2%} (🔺+{diff:.2%})"
    elif diff < 0: return f"{new_acc:.2%} (🔻{diff:.2%})"
    else: return f"{new_acc:.2%} (➖)"

def save_report_to_file(lines, save_path):
    """保存文本报告"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def run_detailed_eval(model, tokenizer,device,test_path=test_path):
    """
    执行详细评估的主函数
    :param model: 训练好的模型对象
    :param tokenizer: 分词器
    :param device: 运行设备
    """
    log_dir = Path(cfg["model"]["log_dir"])
    
    print(f"\n🔎 正在对测试集进行【细粒度能力体检】...")
    
    # 1. 读取原始CSV（因为我们需要 type 和 domain 列，Dataset对象里可能已经被处理掉了）
    df = pd.read_csv(test_path)
    texts = df["text"].tolist()
    
    # 2. 批量推理
    model.eval()
    all_preds = []
    batch_size = 32
    
    # 使用 tqdm 显示进度
    for i in tqdm(range(0, len(texts), batch_size), desc="推理中", leave=False):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            
    df["pred"] = all_preds
    df["is_correct"] = df["pred"] == df["label"]
    
    # 3. 生成报告内容
    report_lines = []
    report_lines.append("="*60)
    report_lines.append(f"📊 模型能力详细评估报告 (Run: {cfg['train'].get('run_name')})")
    report_lines.append("="*60)
    
    # --- 维度1：逻辑句式 ---
    report_lines.append("\n【1. 逻辑句式能力 (Logic Type)】")
    report_lines.append(f"{'类型':<10} | {'样本':<5} | {'基准':<8} | {'当前 (变化)'}")
    report_lines.append("-" * 50)
    for type_, score in df.groupby("type")["is_correct"].mean().items():
        base = BASELINE["logic"].get(type_, 0)
        report_lines.append(f"{type_:<10} | {len(df[df['type']==type_]):<5} | {base:.2%} | {print_diff(score, base)}")

    # --- 维度2：垂直领域 ---
    report_lines.append("\n【2. 垂直领域能力 (Domain)】")
    report_lines.append(f"{'领域':<10} | {'样本':<5} | {'基准':<8} | {'当前 (变化)'}")
    report_lines.append("-" * 50)
    for dom, score in df.groupby("domain")["is_correct"].mean().items():
        base = BASELINE["domain"].get(dom, 0)
        report_lines.append(f"{dom:<10} | {len(df[df['domain']==dom]):<5} | {base:.2%} | {print_diff(score, base)}")

    # --- 维度3：全局 ---
    total_acc = df["is_correct"].mean()
    report_lines.append("\n" + "="*60)
    report_lines.append(f"🏆 全局准确率 (Overall Accuracy): {total_acc:.2%}")
    report_lines.append("="*60)

    # 4. 打印并保存
    report_str = "\n".join(report_lines)
    print(report_str) # 打印到控制台
    
    # 保存报告文本
    report_path = log_dir / "detailed_report.txt"
    save_report_to_file(report_lines, report_path)
    
    # 保存错题本 (Bad Cases)
    bad_cases = df[~df["is_correct"]]
    bad_case_path = log_dir / "bad_cases.csv"
    bad_cases.to_csv(bad_case_path, index=False, encoding="utf-8-sig")
    
    print(f"\n📝 详细报告已保存至: {report_path}")
    print(f"❌ 错题本已保存至: {bad_case_path} (共 {len(bad_cases)} 条错误)")