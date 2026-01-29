import pandas as pd
from sklearn.model_selection import train_test_split

# ===================== 固定配置项（7:2:1比例，贴合你的数据，无需修改）=====================
# 最终清洗整合后的干净数据文件
INPUT_CSV_PATH = "data/processed/doubao_generated_final.csv"
# 拆分后文件保存路径（统一放在processed目录，方便微调调用）
TRAIN_SAVE_PATH = "data/train/train.csv"
VAL_SAVE_PATH = "data/train/val.csv"
TEST_SAVE_PATH = "data/train/test.csv"
# 拆分比例：7:2:1（训练集:验证集:测试集），已精准配置
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
# 分层抽样依据：按type（句式）+ label（情感标签）分层，保证分布1:1还原
STRATIFY_COLS = ["type", "label"]
# 随机种子：固定为42，保证拆分结果可复现（每次运行拆分结果完全一致）
RANDOM_SEED = 42
# ================================================================================

def split_train_val_test():
    """
    核心功能：
    1. 加载2169条标准化干净数据，校验核心字段
    2. 两步分层抽样：先拆70%训练集，再从剩余30%中拆2:1的验证/测试集
    3. 验证拆分后各集合的句式、标签分布与原数据一致
    4. 生成可直接用于BERT微调的train/val/test.csv（utf-8-sig无乱码）
    """
    print("🚀 启动训练/验证/测试集分层抽样拆分流程（7:2:1）")
    print(f"{'='*90}")
    print(f"📌 拆分比例：训练集{TRAIN_RATIO*100}% | 验证集{VAL_RATIO*100}% | 测试集{TEST_RATIO*100}%")
    print(f"📌 分层依据：{STRATIFY_COLS}（保证句式+情感标签分布与原数据完全一致）")
    print(f"📌 随机种子：{RANDOM_SEED}（拆分结果可复现，便于后续微调复现实验）")
    print(f"{'='*90}")

    # 步骤1：加载干净数据并校验文件/字段完整性
    if not pd.io.common.file_exists(INPUT_CSV_PATH):
        raise FileNotFoundError(f"干净数据文件不存在：{INPUT_CSV_PATH}，请检查路径")
    df = pd.read_csv(INPUT_CSV_PATH, encoding='utf-8-sig')
    print(f"📁 加载成功 | 原始干净数据 | 总样本数：{len(df)}条")

    # 校验分层字段是否存在，避免拆分失败
    missing_stratify_cols = [col for col in STRATIFY_COLS if col not in df.columns]
    if missing_stratify_cols:
        raise ValueError(f"数据缺失分层字段：{missing_stratify_cols}，请检查文件格式")

    # 步骤2：核心-两步分层抽样拆分（保证7:2:1精准比例+分布一致）
    # 第一步：从总数据中拆出70%训练集，剩余30%为临时集（验证+测试）
    df_train, df_temp = train_test_split(
        df,
        test_size=VAL_RATIO + TEST_RATIO,  # 临时集占比30%
        random_state=RANDOM_SEED,
        stratify=df[STRATIFY_COLS]  # 按句式+标签分层，第一次保证训练集分布一致
    )
    # 第二步：从30%临时集中拆出2:1的验证集和测试集（精准匹配20%:10%总数据占比）
    df_val, df_test = train_test_split(
        df_temp,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),  # 测试集占临时集1/3，即总数据10%
        random_state=RANDOM_SEED,
        stratify=df_temp[STRATIFY_COLS]  # 第二次分层，保证验证/测试集分布一致
    )

    # 步骤3：重置索引（拆分后索引混乱，重置为连续0开始，适配BERT微调加载逻辑）
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # 步骤4：保存拆分后文件（utf-8-sig编码，避免Windows/Mac中文乱码，不保留索引）
    df_train.to_csv(TRAIN_SAVE_PATH, index=False, encoding='utf-8-sig')
    df_val.to_csv(VAL_SAVE_PATH, index=False, encoding='utf-8-sig')
    df_test.to_csv(TEST_SAVE_PATH, index=False, encoding='utf-8-sig')

    # 步骤5：打印拆分后各集合样本数统计
    print(f"\n📊 分层抽样拆分完成！各集合样本数（精准7:2:1）：")
    train_pct = round(len(df_train)/len(df)*100, 2)
    val_pct = round(len(df_val)/len(df)*100, 2)
    test_pct = round(len(df_test)/len(df)*100, 2)
    print(f"  - 训练集（train.csv）：{len(df_train)}条（{train_pct}%）")
    print(f"  - 验证集（val.csv）：{len(df_val)}条（{val_pct}%）")
    print(f"  - 测试集（test.csv）：{len(df_test)}条（{test_pct}%）")
    print(f"  - 合计：{len(df_train)+len(df_val)+len(df_test)}条（与原始数据完全一致，无丢失）")
    print(f"{'='*90}")

    # 步骤6：分布一致性验证（打印核心字段占比，直观确认无偏差）
    print(f"📈 拆分后核心字段分布验证（与原数据一致即为合格）")
    # 定义统一的占比打印函数，方便对比
    def print_distribution(data, name):
        print(f"  ▶ {name} - 句式类型（type）占比：")
        type_dist = data["type"].value_counts(normalize=True) * 100
        for t, ratio in type_dist.items():
            print(f"     • {t}：{round(ratio, 2)}%")
        print(f"  ▶ {name} - 情感标签（label）占比：")
        label_dist = data["label"].value_counts(normalize=True) * 100
        for l, ratio in label_dist.items():
            print(f"     • label={l}：{round(ratio, 2)}%")
        print(f"  ————————————————————")

    # 依次打印原数据、训练集、验证集、测试集的分布，方便对比
    print_distribution(df, "原始数据")
    print_distribution(df_train, "训练集")
    print_distribution(df_val, "验证集")
    print_distribution(df_test, "测试集")

    # 步骤7：打印最终文件路径，方便后续微调调用
    print(f"💾 拆分后文件保存路径（可直接用于BERT微调）：")
    print(f"  - 训练集：{TRAIN_SAVE_PATH}")
    print(f"  - 验证集：{VAL_SAVE_PATH}")
    print(f"  - 测试集：{TEST_SAVE_PATH}")
    print(f"{'='*90}")
    print(f"🎉 7:2:1分层抽样拆分完成！三个文件可直接投入BERT微调～")

if __name__ == "__main__":
    try:
        split_train_val_test()
    except Exception as e:
        print(f"\n❌ 程序执行出错：{str(e)}")
        print(f"💡 快速排查：1. 检查文件路径是否正确 2. 确认CSV文件未损坏 3. 验证字段是否完整")