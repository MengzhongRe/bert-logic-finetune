# scripts/evaluator.py
import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
# ✅ 引入必要的 sklearn 指标库
from sklearn.metrics import accuracy_score, f1_score, classification_report

def save_report_to_file(lines, save_path):
    """保存文本报告"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def calculate_metrics(df_subset, pred_col='pred', label_col='label'):
    """
    ✅ 修改后的核心指标计算函数：
    返回: Accuracy, Macro-F1, Recall_0(负面), Recall_1(正面), Count
    """
    if len(df_subset) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    y_true = df_subset[label_col]
    y_pred = df_subset[pred_col]

    # 1. 全局准确率 (Accuracy)
    acc = accuracy_score(y_true, y_pred)
    
    # 2. Macro-F1 (宏平均F1，更能反映模型综合能力，防止偏科)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 3. 获取详细分报告以提取 Recall_0 和 Recall_1
    # output_dict=True 返回字典格式
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # 提取负面(0)和正面(1)的召回率
    # 字典的key通常是字符串 '0' 和 '1'
    rec_0 = report.get('0', {}).get('recall', 0.0)
    rec_1 = report.get('1', {}).get('recall', 0.0)

    return acc, macro_f1, rec_0, rec_1, len(df_subset)

def run_detailed_eval(model, tokenizer, device, test_path, output_dir='eval_results'):
    """
    执行详细评估的主函数 (已适配新指标)
    """
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔎 正在对数据集进行评估: {test_path}")
    
    # 1. 读取数据
    if not os.path.exists(test_path):
        print(f"❌ 错误：找不到测试文件 {test_path}")
        return

    df = pd.read_csv(test_path)
    # 确保标签是int类型
    df['label'] = df['label'].astype(int)
    texts = df["text"].tolist()
    
    # 2. 批量推理
    model.eval()
    all_preds = []
    batch_size = 32
    
    for i in tqdm(range(0, len(texts), batch_size), desc="推理中", leave=False):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            
    df["pred"] = all_preds
    df["is_correct"] = df["pred"] == df["label"]
    
    # 3. 计算全局指标 (✅ 这里解包 5 个变量)
    total_acc, total_macro_f1, total_rec0, total_rec1, total_count = calculate_metrics(df)
    
    # --- 生成报告内容 ---
    report_lines = []
    report_lines.append("="*60)
    report_lines.append(f"📊 模型评估报告 | 模型: uer/roberta-base-finetuned-dianping-chinese")
    report_lines.append(f"📂 数据集: {test_path}")
    report_lines.append("="*60)
    
    # ✅ 更新全局报告文本
    report_lines.append("\n【1. 全局指标 (Global Metrics)】")
    report_lines.append(f"样本总数: {total_count}")
    report_lines.append(f"准确率 (Accuracy)   : {total_acc:.2%}")
    report_lines.append(f"宏平均F1 (Macro-F1) : {total_macro_f1:.2%}")
    report_lines.append(f"负面召回 (Recall_0) : {total_rec0:.2%} (正话反说/双重否定能力)")
    report_lines.append(f"正面召回 (Recall_1) : {total_rec1:.2%} (反话正说能力)")

    # --- 维度2：分句式能力 (By Sentence Type) ---
    report_lines.append("\n【2. 句式维度评估 (By Logic Type)】")
    type_stats = []
    for type_, group in df.groupby("type"):
        # ✅ 循环内解包 5 个变量
        acc, mf1, rec0, rec1, count = calculate_metrics(group)
        type_stats.append({
            "句式": type_, 
            "样本数": count, 
            "准确率": acc, 
            "Macro-F1": mf1,
            "负面召回": rec0,
            "正面召回": rec1
        })
    
    # 转为DataFrame
    df_type_stats = pd.DataFrame(type_stats).sort_values(by="Macro-F1", ascending=False)
    report_lines.append(df_type_stats.to_markdown(index=False, floatfmt=".2%"))

    # --- 维度3：分领域能力 (By Domain) ---
    report_lines.append("\n【3. 领域维度评估 (By Domain)】")
    domain_stats = []
    for dom, group in df.groupby("domain"):
        # ✅ 循环内解包 5 个变量
        acc, mf1, rec0, rec1, count = calculate_metrics(group)
        domain_stats.append({
            "领域": dom, 
            "样本数": count, 
            "准确率": acc, 
            "Macro-F1": mf1,
            "负面召回": rec0,
            "正面召回": rec1
        })
    df_domain_stats = pd.DataFrame(domain_stats).sort_values(by="Macro-F1", ascending=False)
    report_lines.append(df_domain_stats.to_markdown(index=False, floatfmt=".2%"))

    # --- 维度4：句式-领域 交叉评估 (By Type-Domain Pair) ---
    report_lines.append("\n【4. 句式-领域 交叉评估 (Type-Domain Pair)】")
    pair_stats = []
    for (type_, dom), group in df.groupby(["type", "domain"]):
        # ✅ 循环内解包 5 个变量
        acc, mf1, rec0, rec1, count = calculate_metrics(group)
        pair_stats.append({
            "句式": type_, 
            "领域": dom, 
            "样本数": count, 
            "准确率": acc, 
            "Macro-F1": mf1,
            "负面召回": rec0,
            "正面召回": rec1
        })
    df_pair_stats = pd.DataFrame(pair_stats).sort_values(by=["句式", "Macro-F1"], ascending=[True, False])
    
    report_lines.append("(完整表格请查看生成的 metrics_by_pair.csv 文件)")
    report_lines.append(df_pair_stats.head(10).to_markdown(index=False, floatfmt=".2%"))
    
    # 4. 打印并保存
    report_str = "\n".join(report_lines)
    print(report_str) 
    
    # --- 保存各类文件 ---
    save_report_to_file(report_lines, output_path / "final_report.txt")
    
    df_type_stats.to_csv(output_path / "metrics_by_type.csv", index=False, encoding='utf-8-sig')
    df_domain_stats.to_csv(output_path / "metrics_by_domain.csv", index=False, encoding='utf-8-sig')
    df_pair_stats.to_csv(output_path / "metrics_by_pair.csv", index=False, encoding='utf-8-sig')
    
    # 错题本保持不变
    bad_cases = df[~df["is_correct"]]
    bad_cases.to_csv(output_path / "bad_cases.csv", index=False, encoding="utf-8-sig")
    
    print(f"\n✅ 评估完成！结果已保存至 '{output_dir}' 目录")