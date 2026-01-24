import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
import wandb
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ================= 1. WandB 初始化 =================
wandb.init(project="logic_sentiment_test", entity="mengzhongren-sun-yat-sen-university", name="bert_logic_eval")

# ================= 2. 配置环境 =================
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")

# ================= 3. 加载模型 =================
model_checkpoint = "uer/roberta-base-finetuned-dianping-chinese"
print(f"正在加载模型: {model_checkpoint}...")
classifier = pipeline("sentiment-analysis", model=model_checkpoint, device=device)

# ================= 4. 读取数据 =================
csv_path = "data/deepseek_structured_data_v2.csv"
try:
    df = pd.read_csv(csv_path)
    print(f"✅ 成功加载数据集，共 {len(df)} 条。")
except FileNotFoundError:
    print("❌ 找不到文件，生成示例数据")
    df = pd.DataFrame([{"text": "演示数据", "label": "Positive", "type": "简单句"}])

# ================= 5. 标签映射函数 =================
def map_prediction(pred_result):
    label = pred_result['label']
    if 'positive' in label.lower():
        return "Positive"
    elif 'negative' in label.lower():
        return "Negative"

# ================= 6. 批量推理 =================
print("🚀 开始推理...")
predictions = []
confidences = []

for text in tqdm(df["text"].tolist()):
    pred = classifier(text, truncation=True, max_length=512)[0]
    mapped_label = map_prediction(pred)
    predictions.append(mapped_label)
    confidences.append(pred['score'])

df["Pred_Label"] = predictions
df["Confidence"] = confidences
df["Is_Correct"] = df["label"] == df["Pred_Label"]

# ================= 7. 生成统计报告 =================
report = df.groupby("type")["Is_Correct"].mean().reset_index()
report.columns = ["句式类型", "准确率"]
report["准确率"] = report["准确率"].apply(lambda x: x*100)

# 上传统计结果到 WandB
for _, row in report.iterrows():
    wandb.log({f"accuracy/{row['句式类型']}": row["准确率"]})

print("\n📊 逻辑陷阱能力评估报告")
print(report.to_markdown(index=False))
report.to_csv('result/accuracy_report_v2.csv',index=False)

# ================= 8. 保存预测结果 =================
os.makedirs("result", exist_ok=True)
output_file = "result/logic_test_result_v2.csv"
df.to_csv(output_file, index=False)
print(f"💾 详细结果已保存至: {output_file}")
wandb.save(output_file)

# ================= 10. Bad Case 分析 =================
bad_cases = df[df["Is_Correct"] == False]
for t in df["type"].unique():
    t_bad = bad_cases[bad_cases["type"] == t].head(10)  # 每类取前10条
    if not t_bad.empty:
        wandb.log({f"bad_cases/{t}": wandb.Table(data=t_bad, columns=t_bad.columns.tolist())})

print("✅ 所有结果已同步到 WandB！")
