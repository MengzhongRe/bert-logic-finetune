import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
import wandb
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib
import warnings

# 屏蔽所有字体相关警告和Seaborn绘图警告
warnings.filterwarnings('ignore', category=UserWarning)
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimSun', 'WenQuanYi Zen Hei', 'Heiti TC']  # 兜底字体，Linux必带
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.family'] = 'sans-serif'
# 屏蔽Matplotlib字体查找的冗余日志（关键：消除findfont警告
sns.set(rc={'figure.figsize': (10, 6)}, font='sans-serif')  # Seaborn不用指定具体中文字体，用兜底即可

NUMBER = 3
PATH = f'data/test_result/v{NUMBER}'
csv_path = f"data/raw/deepseek_structured_data_cyclic_domain_960_v{NUMBER}.csv"

# ================= 1. WandB 初始化（增加备注，方便后续对比） =================
wandb.init(
    project="logic_sentiment_test", 
    entity="mengzhongren-sun-yat-sen-university", 
    name="bert_logic_eval_base",  # 后缀加base，区分微调后的模型
    notes="uer/roberta-base-finetuned-dianping-chinese 基准模型评估，按句式/领域双维度统计",
    mode='offline'
)

# ================= 2. 配置环境（增加GPU信息打印，方便排查） =================
device = 0 if torch.cuda.is_available() else -1
gpu_info = torch.cuda.get_device_name(0) if device == 0 else "CPU"
print(f"Using device: {gpu_info} (cuda:0)" if device == 0 else "Using device: CPU")
# 固定随机种子，保证结果可复现
torch.manual_seed(42)

# ================= 3. 加载模型（增加pipeline参数优化，提升推理效率） =================
model_checkpoint = "uer/roberta-base-finetuned-dianping-chinese"
print(f"正在加载模型: {model_checkpoint}...")
# 优化pipeline参数：指定padding、批量推理（num_workers）、关闭进度条
classifier = pipeline(
    "sentiment-analysis", 
    model=model_checkpoint, 
    device=device,
    padding="max_length",  # 统一padding，避免逐句处理的开销
    truncation=True,
    max_length=512,
    num_workers=os.cpu_count(),  # 利用多核CPU处理数据加载
    batch_size=32,  # 批量推理，大幅提升速度（GPU建议32/64，CPU建议8/16）
)

# ================= 4. 读取数据（增加数据校验，避免空数据/字段缺失） =================
required_cols = ["text", "label", "type", "domain"]  # 核心字段校验
try:
    df = pd.read_csv(csv_path)
    # 校验核心字段是否存在
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据集缺失核心字段：{missing_cols}，请检查数据格式")
    # 去重（避免重复样本影响准确率）
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"✅ 成功加载并清洗数据集，共 {len(df)} 条有效样本。")
except FileNotFoundError:
    print("❌ 找不到文件，生成示例数据")
    df = pd.DataFrame([
        {"text": "演示数据：这家火锅味道超正宗", "label": "Positive", "type": "简单句", "domain": "美食餐饮"},
        {"text": "演示数据：这部剧剧情超无聊", "label": "Negative", "type": "简单句", "domain": "影视娱乐"}
    ])
except Exception as e:
    print(f"❌ 数据加载失败：{str(e)}")
    exit()

# ================= 5. 标签映射函数（提升健壮性，处理未知标签） =================
def map_prediction(pred_result):
    label = pred_result['label'].lower()
    if 'positive' in label:
        return "Positive"
    elif 'negative' in label:
        return "Negative"
    else:
        # 处理模型输出未知标签的情况，标记为错误预测
        print(f"⚠️  检测到未知标签：{pred_result['label']}，标记为Negative")
        return "Unknown"

# ================= 6. 批量推理（优化效率，处理批量输出） =================
print("🚀 开始批量推理...")
# 提取文本列表，利用pipeline的批量推理能力
text_list = df["text"].tolist()
# 批量推理，tqdm监控整体进度
pred_results = list(tqdm(classifier(text_list), total=len(text_list), desc="推理进度"))

# 解析预测结果
predictions = [map_prediction(pred) for pred in pred_results]
confidences = [pred['score'] for pred in pred_results]

# 新增：标记未知标签的预测，方便后续统计
df["Pred_Label"] = predictions
df["Confidence"] = confidences
# 处理Unknown标签为错误预测
df["Is_Correct"] = (df["label"] == df["Pred_Label"]) & (df["Pred_Label"] != "Unknown")

# ================= 7. 生成精细化统计报告（核心优化：双维度统计+分类报告） =================
print("\n📊 【核心】逻辑句式能力评估报告")
# 7.1 按句式类型统计（原需求，保留并优化）
type_report = df.groupby("type")["Is_Correct"].agg(['mean', 'count']).reset_index()
type_report.columns = ["句式类型", "准确率", "样本数"]
type_report["准确率"] = type_report["准确率"].apply(lambda x: round(x*100, 2))  # 保留2位小数，更直观
print(type_report.to_markdown(index=False, floatfmt=".2f"))

# 7.2 新增：按领域统计准确率（补充维度，看模型在不同领域的表现）
print("\n🌐 【补充】各领域能力评估报告")
domain_report = df.groupby("domain")["Is_Correct"].agg(['mean', 'count']).reset_index()
domain_report.columns = ["领域", "准确率", "样本数"]
domain_report["准确率"] = domain_report["准确率"].apply(lambda x: round(x*100, 2))
print(domain_report.to_markdown(index=False, floatfmt=".2f"))

# 7.3 新增：按「句式+领域」交叉统计（精准定位短板，比如「反讽+日常社交」表现差）
print("\n🔍 【精准】句式+领域交叉准确率（前20条，按准确率升序）")
cross_report = df.groupby(["type", "domain"])["Is_Correct"].agg(['mean', 'count']).reset_index()
cross_report.columns = ["句式类型", "领域", "准确率", "样本数"]
cross_report["准确率"] = cross_report["准确率"].apply(lambda x: round(x*100, 2))
cross_report = cross_report.sort_values("准确率", ascending=True).head(20)  # 优先看表现差的组合
print(cross_report.to_markdown(index=False, floatfmt=".2f"))

# 7.4 新增：全局分类报告（精确率、召回率、F1，比单纯准确率更全面）
print("\n📈 【全局】分类性能报告（Precision/Recall/F1）")
# 过滤Unknown标签，避免影响统计
valid_df = df[df["Pred_Label"] != "Unknown"]
if len(valid_df) > 0:
    global_report = classification_report(
        valid_df["label"], 
        valid_df["Pred_Label"], 
        target_names=["Negative", "Positive"],
        output_dict=True,
        digits=2
    )
    # 打印全局报告
    print(classification_report(
        valid_df["label"], 
        valid_df["Pred_Label"], 
        target_names=["Negative", "Positive"],
        digits=2
    ))
else:
    global_report = {}
    print("⚠️  无有效预测样本，无法生成全局分类报告")

# ================= 8. WandB日志优化（多维度日志+可视化图表） =================
# 8.1 日志各维度准确率（句式+领域+全局）
# 句式准确率
for _, row in type_report.iterrows():
    wandb.log({f"accuracy/按句式/{row['句式类型']}": row["准确率"]})
# 领域准确率
for _, row in domain_report.iterrows():
    wandb.log({f"accuracy/按领域/{row['领域']}": row["准确率"]})
# 全局准确率
global_accuracy = round(df["Is_Correct"].mean()*100, 2)
wandb.log({"accuracy/全局准确率": global_accuracy})
wandb.log({"data/有效样本数": len(df)})
wandb.log({"data/错误样本数": len(df) - df["Is_Correct"].sum()})

# 8.2 日志全局分类指标（P/R/F1）
if global_report:
    wandb.log({
        "metric/negative_precision": global_report["Negative"]["precision"],
        "metric/negative_recall": global_report["Negative"]["recall"],
        "metric/negative_f1": global_report["Negative"]["f1-score"],
        "metric/positive_precision": global_report["Positive"]["precision"],
        "metric/positive_recall": global_report["Positive"]["recall"],
        "metric/positive_f1": global_report["Positive"]["f1-score"],
        "metric/weighted_f1": global_report["weighted avg"]["f1-score"]
    })

# 8.3 生成并日志混淆矩阵（可视化模型正负样本的预测偏差）
if len(valid_df) > 0:
    cm = confusion_matrix(valid_df["label"], valid_df["Pred_Label"], labels=["Negative", "Positive"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["预测Negative", "预测Positive"],
                yticklabels=["真实Negative", "真实Positive"],
                annot_kws={"size": 12})
    plt.title(f"Confusion Matrix (Global Accuracy: {global_accuracy}%)", fontsize=14)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    wandb.log({"visual/confusion_matrix": wandb.Image(plt)})
    plt.close()

# ================= 9. 保存预测结果（保存所有报告，方便本地分析） =================
os.makedirs(PATH, exist_ok=True)
# 保存详细预测结果
output_file = f"{PATH}/logic_test_result.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")  # utf-8-sig避免中文乱码
# 保存各维度统计报告
type_report.to_csv(f"{PATH}/accuracy_report_by_type.csv", index=False, encoding="utf-8-sig")
domain_report.to_csv(f"{PATH}/accuracy_report_by_domain.csv", index=False, encoding="utf-8-sig")
cross_report.to_csv(f"{PATH}/accuracy_report_cross.csv", index=False, encoding="utf-8-sig")
if global_report:
    pd.DataFrame(global_report).T.to_csv(f"{PATH}/global_classification_report.csv", encoding="utf-8-sig")

print(f"\n💾 所有结果已保存至{PATH}目录：")
print(f"   - 详细预测结果：{output_file}")
print(f"   - 句式准确率报告：{PATH}/accuracy_report_by_type.csv")
print(f"   - 全局分类报告：{PATH}/global_classification_report.csv")
# WandB保存关键文件
wandb.save(f"{PATH}/*.csv")

# ================= 10. Bad Case 分析优化（精细化分组+特征标注） =================
bad_cases = df[df["Is_Correct"] == False].reset_index(drop=True)
if not bad_cases.empty:
    print(f"\n❌ 共检测到 {len(bad_cases)} 条错误样本（Bad Case），占比 {round(len(bad_cases)/len(df)*100, 2)}%")
    # 10.1 按句式分组日志到WandB，方便在线查看
    for t in df["type"].unique():
        t_bad = bad_cases[bad_cases["type"] == t]
        if not t_bad.empty:
            wandb.log({f"bad_cases/按句式/{t}": wandb.Table(data=t_bad, columns=t_bad.columns.tolist())})
    # 10.2 按领域分组日志到WandB
    for d in df["domain"].unique():
        d_bad = bad_cases[bad_cases["domain"] == d]
        if not d_bad.empty:
            wandb.log({f"bad_cases/按领域/{d}": wandb.Table(data=d_bad, columns=d_bad.columns.tolist())})
    # 10.3 保存所有Bad Case到本地，方便手动分析
    bad_cases.to_csv(f"{PATH}/bad_cases.csv", index=False, encoding="utf-8-sig")
    wandb.save(f"{PATH}/bad_cases.csv")
else:
    print("\n✅ 无错误样本，模型在该数据集上表现完美！")

# ================= 11. 结束WandB运行 =================
wandb.finish()
print("\n✅ 所有评估流程完成，结果已同步到WandB，本地文件已保存！")