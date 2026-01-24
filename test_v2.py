import pandas as pd
from transformers import pipeline
from tqdm import tqdm # 进度条库
import torch

# 1. 配置环境
# 检测是否有显卡，有的话用显卡跑，速度快 10 倍
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")

# 2. 加载模型
# 继续使用 UER 的这个中文情感模型，它在中文社区很常用，适合做 Baseline
model_checkpoint = "uer/roberta-base-finetuned-dianping-chinese"
print(f"正在加载模型: {model_checkpoint}...")
classifier = pipeline("sentiment-analysis", model=model_checkpoint, device=device)

# 3. 读取你刚才生成的数据
csv_path = "data/deepseek_structured_data_v1.csv"
try:
    df = pd.read_csv(csv_path)
    print(f"✅ 成功加载数据集，共 {len(df)} 条。")
except FileNotFoundError:
    print("❌ 找不到文件，请确认 logic_stress_test_v2.csv 是否在当前目录下！")
    # 如果找不到文件，为了演示代码，这里生成一个伪造的小 dataframe
    df = pd.DataFrame([{"text": "演示数据", "label": "Positive", "type": "简单"}])

# 4. 定义标签映射函数 (关键步骤)
# 不同的模型输出的 label 不一样，有的输出 'positive'，有的输出 'star 5'
# 我们需要把它统一映射成你的 CSV 里的 'Positive' / 'Negative'
def map_prediction(pred_result):
    label = pred_result['label']
    # 针对 uer/roberta-base-finetuned-dianping-chinese 的输出处理
    # 这个模型通常输出: 'positive' (对应5星), 'negative' (对应1星)
    # 或者 label_0 / label_1
    
    # 宽松匹配逻辑
    if 'positive' in label:
        return "Positive"
    elif 'negative' in label:
        return "Negative"

# 5. 开始批量推理
print("🚀 开始推理...")
predictions = []
confidences = []

# 使用 tqdm 显示进度条，看起来更专业
for text in tqdm(df["text"].tolist()):
    # 截断长度，防止爆显存（虽然这些句子都很短）
    pred = classifier(text, truncation=True, max_length=512)[0]
    mapped_label = map_prediction(pred)
    
    predictions.append(mapped_label)
    confidences.append(pred['score'])

# 将结果写回 DataFrame
df["Pred_Label"] = predictions
df["Confidence"] = confidences
df["Is_Correct"] = df["label"] == df["Pred_Label"]

# 6. 生成统计报告 (面试要展示的表格)
print("\n" + "="*40)
print("📊 逻辑陷阱能力评估报告")
print("="*40)

# 按“类型”分组计算准确率
report = df.groupby("type")["Is_Correct"].mean().reset_index()
report.columns = ["句式类型", "准确率"]
# 把准确率转换成百分比格式
report["准确率"] = report["准确率"].apply(lambda x: f"{x*100:.2f}%")

# 使用 Markdown 格式打印，方便复制
print(report.to_markdown(index=False))

# 7. 保存带有预测结果的文件 (用于 Bad Case 分析)
output_file = "result/logic_test_result_final.csv"
df.to_csv(output_file, index=False)
print(f"\n💾 详细结果已保存至: {output_file}")

# 8. 抽出几个典型的 Bad Case 看看
print("\n🔍 典型 Bad Case 展示 (反讽类型):")
bad_cases = df[(df["type"] == "反讽") & (df["Is_Correct"] == False)].head(3)
for _, row in bad_cases.iterrows():
    print(f"文本: {row['text']}")
    print(f"真实: {row['label']} | 预测: {row['Pred_Label']} | 置信度: {row['Confidence']:.4f}")
    print("-" * 30)