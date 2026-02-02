# scripts/prepare_data_stratified.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# ================= 配置 =================
BASE_PATH = 'data/processed/base_data.csv'      # 基础集
BOOST_PATH = 'data/processed/aug_data.csv' # 增强集

# 输出路径
TRAIN_OUTPUT_PATH = 'data/train/train.csv'
VAL_OUTPUT_PATH = 'data/train/val.csv'
# ========================================

test_size = 0.2

def prepare_stratified_data():
    print("🔥 开始执行【多维分层】数据熔炼与切分...")
    
    # 1. 读取并合并
    if not os.path.exists(BASE_PATH) or not os.path.exists(BOOST_PATH):
        print("❌ 错误：文件缺失。")
        return

    df_base = pd.read_csv(BASE_PATH)
    df_boost = pd.read_csv(BOOST_PATH)
    df_all = pd.concat([df_base, df_boost], ignore_index=True)
    
    # 清洗：去重 + 去异常 + 去na
    df_all = df_all.drop_duplicates(subset=['text'],keep='first')
    df_all = df_all[
        (df_all['text'].str.len() >= 6) &
        (df_all['text'].str.len() <= 35)
    ]
    df_all = df_all.dropna(subset=['text', 'label', 'type', 'domain'])
    df_all['label'] = df_all['label'].astype(int)
    
    print(f"   - 总样本数: {len(df_all)}")
    
    # =======================================================
    # 2. 核心步骤：构建组合分层键 (Composite Stratification Key)
    # =======================================================
    # 目标：同时兼顾 句式(Type) + 情感(Label) + 领域(Domain)
    # 组合键格式示例: "反讽_0_影视娱乐"
    
    df_all['stratify_key'] = (
        df_all['type'].astype(str) + "_" + 
        df_all['label'].astype(str) + "_" + 
        df_all['domain'].astype(str)
    )
    
    # ⚠️ 安全检查：防止某个极度稀有的组合只出现 1 次
    # 如果某组合只有 1 条数据，无法拆分到 Train 和 Val，这会导致报错。
    # 策略：如果 Count < 2，则回退到只按 "句式_情感" 分层
    key_counts = df_all['stratify_key'].value_counts()
    singletons = key_counts[key_counts < 2].index
    
    if len(singletons) > 0:
        print(f"⚠️ 发现 {len(singletons)} 个稀有组合仅有1条样本，将进行模糊分层处理...")
        def fallback_key(row):
            full_key = f"{row['type']}_{row['label']}_{row['domain']}"
            if full_key in singletons:
                # 回退策略：去掉 domain，只保住句式和情感
                return f"{row['type']}_{row['label']}"
            return full_key
        
        df_all['stratify_key'] = df_all.apply(fallback_key, axis=1)
    
    # 3. 执行分层切分
    train_df, val_df = train_test_split(
        df_all, 
        test_size=test_size,    # 10% 验证集
        random_state=42,  
        shuffle=True,     
        stratify=df_all['stratify_key'] # ✅ 这里传入组合键
    )
    
    # 4. 移除临时列并保存
    train_df = train_df.drop(columns=['stratify_key'])
    val_df = val_df.drop(columns=['stratify_key'])
    
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    val_df.to_csv(VAL_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    print("\n✅ 数据划分完成！")
    print(f"📁 训练集: {len(train_df)} -> {TRAIN_OUTPUT_PATH}")
    print(f"📁 验证集: {len(val_df)} -> {VAL_OUTPUT_PATH}")
    
    # =======================================================
    # 5. 分布一致性校验 (让用户放心)
    # =======================================================
    print("\n📊 分布一致性检查 (验证集 vs 训练集 占比):")
    print("-" * 50)
    print(f"{'维度':<15} | {'Val占比':<10} | {'Train占比':<10} | {'偏差'}")
    print("-" * 50)
    
    def check_dist(col_name):
        val_dist = val_df[col_name].value_counts(normalize=True)
        train_dist = train_df[col_name].value_counts(normalize=True)
        
        for category in val_dist.index:
            v_p = val_dist[category]
            t_p = train_dist.get(category, 0)
            diff = abs(v_p - t_p)
            # 只打印偏差较大的，或者前几个
            print(f"{category:<15} | {v_p:.2%}    | {t_p:.2%}    | {diff:.4f}")

    print(">>> 1. 情感分布 (Label):")
    check_dist('label')
    print("\n>>> 2. 句式分布 (Type):")
    check_dist('type')
    print("\n>>> 3. 领域分布 (Domain, Top 3):") # 领域太多，只看前3
    val_domain_counts = val_df['domain'].value_counts(normalize=True).head(3)
    for dom in val_domain_counts.index:
         v_p = val_domain_counts[dom]
         t_p = train_df['domain'].value_counts(normalize=True).get(dom, 0)
         print(f"{dom:<15} | {v_p:.2%}    | {t_p:.2%}    | {abs(v_p-t_p):.4f}")

if __name__ == "__main__":
    prepare_stratified_data()