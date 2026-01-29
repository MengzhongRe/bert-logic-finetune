import os
import pandas as pd

# ===================== 固定配置项（无需修改，贴合你的两个文件）=====================
# 豆包生成的数据集（已用方案二修改过type=转折→转折关系）
CORE_CSV_PATH = "data/processed/doubao_generated_630_samples.csv"
# 原有高质量预处理数据集（含简单句、转折关系）
OTHER_CSV_PATH = "data/processed/high_acc_base_samples.csv"
# 最终清洗整合后的干净数据保存路径
FINAL_SAVE_PATH = "data/processed/doubao_generated_final.csv"
# 文本长度要求（6-35字，贴合样本生成规则，仅删过短/过长的无效文本）
MIN_TEXT_LEN = 6
MAX_TEXT_LEN = 35
# 核心字段（和你的样本字段完全一致，固定不变）
CORE_COLUMNS = ["text", "label", "type", "sub_type", "domain", "is_valid"]
# =====================================================================================

def data_clean_merge_final():
    """
    最终版数据清洗整合核心函数：
    1. 加载两个源文件，校验字段完整性
    2. 缺失值处理：删除任意核心字段为空的样本（训练会报错）
    3. 文本长度校验：仅保留6-35字的有效文本
    4. 全局去重：按text字段去重（避免同一句话重复，防止模型过拟合）
    5. 字段标准化：统一数据类型、去除字符串空格、保证格式一致
    6. 重置索引：生成连续索引，方便后续微调数据划分
    7. 保存最终干净数据（utf-8-sig编码，兼容Windows/Mac，无中文乱码）
    """
    print("🚀 启动【最终版】数据清洗与整合流程")
    print(f"{'='*90}")

    # 步骤1：加载两个源文件并校验
    df_list = []
    # 加载豆包生成的核心文件
    if not os.path.exists(CORE_CSV_PATH):
        raise FileNotFoundError(f"豆包生成的文件不存在：{CORE_CSV_PATH}，请检查路径")
    df_core = pd.read_csv(CORE_CSV_PATH, encoding='utf-8-sig')
    # 加载原有基础文件
    if not os.path.exists(OTHER_CSV_PATH):
        raise FileNotFoundError(f"原有基础文件不存在：{OTHER_CSV_PATH}，请检查路径")
    df_other = pd.read_csv(OTHER_CSV_PATH, encoding='utf-8-sig')

    # 校验核心字段是否完整，避免缺失字段导致训练报错
    for df, name in zip([df_core, df_other], ["豆包生成数据", "原有基础数据"]):
        missing_cols = [col for col in CORE_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{name}缺失核心字段：{missing_cols}，请检查文件格式")
        df_list.append(df[CORE_COLUMNS].copy())  # 仅保留核心字段，剔除多余字段
        print(f"📁 加载成功 | {name} | 原始样本数：{len(df)}条")

    # 步骤2：合并两个数据集（初步合并，未清洗）
    df_merged = pd.concat(df_list, ignore_index=True)
    print(f"\n📊 初步合并完成 | 两文件总样本数（未清洗）：{len(df_merged)}条")
    print(f"{'='*90}")

    # 步骤3：核心数据清洗（仅删「真坏数据」，不删有效数据）
    print("🔍 开始核心清洗（仅处理无效坏数据，保留所有有效样本）")
    df_clean = df_merged.copy()
    total_original = len(df_clean)

    # 3.1 删除缺失值样本（任意核心字段为空则删除）
    df_clean = df_clean.dropna(subset=CORE_COLUMNS)
    del_na = total_original - len(df_clean)
    print(f"  ▶ 删除缺失值样本：{del_na}条（任意核心字段为空）")

    # 3.2 文本长度校验（6-35字，去除前后空格后计算长度）
    df_clean["text"] = df_clean["text"].astype(str).str.strip()  # 文本去空格
    df_clean["text_len"] = df_clean["text"].apply(len)
    df_clean = df_clean[(df_clean["text_len"] >= MIN_TEXT_LEN) & (df_clean["text_len"] <= MAX_TEXT_LEN)]
    del_len = total_original - del_na - len(df_clean)
    df_clean = df_clean.drop(columns=["text_len"])  # 删除临时长度字段
    print(f"  ▶ 删除长度不符样本：{del_len}条（要求{MIN_TEXT_LEN}-{MAX_TEXT_LEN}字）")

    # 3.3 全局去重（按text字段，保留第一次出现的样本，避免重复）
    df_clean = df_clean.drop_duplicates(subset=["text"], keep="first")
    del_dup = total_original - del_na - del_len - len(df_clean)
    print(f"  ▶ 删除重复样本：{del_dup}条（按text字段全局去重）")

    # 步骤4：字段标准化（统一格式，适配BERT微调，不修改原有字段值含义）
    print(f"\n📐 开始字段标准化（适配BERT微调，统一数据格式）")
    # 4.1 统一数据类型：label为int（模型训练需要数值型）、is_valid为bool，其余为字符串
    df_clean["label"] = df_clean["label"].astype(str).str.strip()
    df_clean = df_clean[df_clean["label"].isin(["0", "1"])]  # 仅保留有效标签
    df_clean["label"] = df_clean["label"].astype(int)

    df_clean["is_valid"] = df_clean["is_valid"].astype(str).str.strip()
    df_clean["is_valid"] = df_clean["is_valid"].map(lambda x: True if x in ["True", "1", "true", "TRUE"] else False)

    # 4.2 所有字符串字段去空格（避免「 转折关系 」和「转折关系」被判定为不同值）
    for col in ["text", "type", "sub_type", "domain"]:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # 4.3 重置索引（合并/去重后索引混乱，重置为连续0开始的索引）
    df_clean = df_clean.reset_index(drop=True)

    # 步骤5：保存最终干净数据
    # 自动创建保存目录（如果不存在）
    save_dir = os.path.dirname(FINAL_SAVE_PATH)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 保存为csv，utf-8-sig编码避免中文乱码，不保留索引
    df_clean.to_csv(FINAL_SAVE_PATH, index=False, encoding='utf-8-sig')

    # 步骤6：打印清洗整合结果统计（清晰看到每一步处理情况）
    print(f"  ▶ 数据类型统一：label(int)、is_valid(bool)、其余字段去空格")
    print(f"  ▶ 索引重置完成：从0开始连续编号")
    print(f"{'='*90}")
    print(f"✅ 清洗整合全部完成！")
    print(f"📊 最终数据统计：")
    print(f"  - 原始总样本数：{total_original}条")
    print(f"  - 累计删除无效样本：{total_original - len(df_clean)}条（缺失值{del_na}+长度不符{del_len}+重复{del_dup}）")
    print(f"  - 最终干净样本数：{len(df_clean)}条")
    print(f"💾 最终文件保存路径：{FINAL_SAVE_PATH}")
    print(f"{'='*90}")

    # 打印样本分布详情（方便你验证数据合理性）
    print(f"📈 最终样本核心分布统计（适配BERT微调分析）")
    print(f"  1. 句式类型（type）分布：")
    type_dist = df_clean["type"].value_counts()
    for t, num in type_dist.items():
        ratio = round(num / len(df_clean) * 100, 2)
        print(f"     - {t}：{num}条（{ratio}%）")
    print(f"  2. 情感标签（label）分布：")
    label_dist = df_clean["label"].value_counts()
    for l, num in label_dist.items():
        ratio = round(num / len(df_clean) * 100, 2)
        print(f"     - label={l}：{num}条（{ratio}%）")
    print(f"  3. 内容领域（domain）分布：")
    domain_dist = df_clean["domain"].value_counts()
    for d, num in domain_dist.items():
        ratio = round(num / len(df_clean) * 100, 2)
        print(f"     - {d}：{num}条（{ratio}%）")
    print(f"  4. 反讽子类型（sub_type）分布（仅反讽样本）：")
    irony_df = df_clean[df_clean["type"] == "反讽"]
    if len(irony_df) > 0:
        irony_sub_dist = irony_df["sub_type"].value_counts()
        for s, num in irony_sub_dist.items():
            ratio = round(num / len(irony_df) * 100, 2)
            print(f"     - {s}：{num}条（{ratio}%）")
    else:
        print(f"     - 无反讽样本")
    print(f"{'='*90}")
    print(f"🎉 所有操作完成！最终文件可直接用于BERT微调～")

if __name__ == "__main__":
    try:
        data_clean_merge_final()
    except Exception as e:
        print(f"\n❌ 程序执行出错：{str(e)}")
        print(f"💡 请检查：1. 文件路径是否正确 2. CSV文件是否损坏 3. 字段是否完整")