import pandas as pd
import random

# ==========================================
# 1. 定义词库 (Lexicon)
# ==========================================
subjects = ["这部电影", "这个设计", "这家餐厅", "这种服务", "这位老师", "这个算法", "这次旅行", "这款游戏", "这个产品", "这次体验"]
pos_adj = ["精彩", "专业", "美味", "周到", "负责", "高效", "愉快", "好玩", "耐用", "难忘"]
neg_adj = ["无聊", "业余", "难吃", "敷衍", "拖沓", "低效", "糟糕", "坑爹", "易坏", "痛苦"]
expensive = ["价格很贵", "排队很久", "位置偏僻", "门票不菲", "等待漫长"]
cheap = ["价格便宜", "不用排队", "交通方便", "免费入场", "响应很快"]

# ==========================================
# 2. 模版生成函数 (Template Generators)
# ==========================================

# A. 双重否定生成器 (Target: Positive)
# 逻辑：不是 + 不 + 正面形容词 = 正面
def gen_double_negation(n=100):
    data = []
    for _ in range(n):
        sub = random.choice(subjects)
        adj = random.choice(pos_adj)
        # 句式1：不是不adj
        # 句式2：不得不说很adj
        # 句式3：没法不喜欢
        templates = [
            f"{sub}不是不{adj}。",
            f"{sub}不得不说是{adj}的。",
            f"对于{sub}，我没法不觉得{adj}。",
            f"我不否认{sub}很{adj}。"
        ]
        data.append({"text": random.choice(templates), "label": "Positive", "type": "双重否定"})
    return data

# B. 转折关系生成器 (Target: Positive/Negative)
# 逻辑：虽然 [负面]，但是 [正面] -> 正面 (侧重后半句)
def gen_concessive(n=100):
    data = []
    for _ in range(n):
        sub = random.choice(subjects)
        p_adj = random.choice(pos_adj)
        n_adj = random.choice(neg_adj)
        bad_condition = random.choice(expensive)
        good_condition = random.choice(cheap)
        
        # 正面转折 (虽然坏，但是好) -> Positive
        s1 = f"虽然{bad_condition}，但是{sub}真的{p_adj}。"
        data.append({"text": s1, "label": "Positive", "type": "转折(正)"})
        
        # 负面转折 (虽然好，但是坏) -> Negative
        s2 = f"虽然{good_condition}，但是{sub}太{n_adj}了。"
        data.append({"text": s2, "label": "Negative", "type": "转折(负)"})
    return data

# C. 简单句生成器 (Baseline)
def gen_simple(n=100):
    data = []
    for _ in range(n):
        sub = random.choice(subjects)
        p_adj = random.choice(pos_adj)
        n_adj = random.choice(neg_adj)
        
        data.append({"text": f"{sub}很{p_adj}。", "label": "Positive", "type": "简单句"})
        data.append({"text": f"{sub}太{n_adj}了。", "label": "Negative", "type": "简单句"})
    return data

# D. 反讽句 (Irony) - 难以用模版生成，手动提供一批高质量的
# 这里给你准备了 20 条，复制扩展即可
irony_data = [
    {"text": "你可真聪明，把系统搞崩了。", "label": "Negative"},
    {"text": "这服务真是太‘棒’了，等了两个小时。", "label": "Negative"},
    {"text": "感谢你在百忙之中敷衍我。", "label": "Negative"},
    {"text": "做得真好，下次别做了。", "label": "Negative"},
    {"text": "这代码写得真艺术，完全看不懂。", "label": "Negative"},
    {"text": "真是谢谢你全家。", "label": "Negative"},
    {"text": "你这建议太有用了，成功浪费了我三天时间。", "label": "Negative"},
    {"text": "这游戏优化得真好，PPT都没这么卡。", "label": "Negative"},
    {"text": "这外卖送得真快，才两个小时就到了。", "label": "Negative"},
    {"text": "真是天才的设计，人类完全没法用。", "label": "Negative"},
    {"text": "我都感动哭了，给我发空包裹。", "label": "Negative"},
    {"text": "你讲得真清楚，我现在更糊涂了。", "label": "Negative"},
    {"text": "这手机续航真强，充电两小时通话五分钟。", "label": "Negative"},
    {"text": "真是良心商家，涨价还减量。", "label": "Negative"},
    {"text": "这牛排真嫩，咬都咬不动。", "label": "Negative"},
    {"text": "这网速快得像蜗牛爬。", "label": "Negative"},
    {"text": "老板真大方，年终奖发了一箱空气。", "label": "Negative"},
    {"text": "真是完美的体验，再也不想来了。", "label": "Negative"},
    {"text": "多亏了你的帮忙，事情变得更糟了。", "label": "Negative"},
    {"text": "这房间真安静，装修队就在隔壁。", "label": "Negative"}
]
# 给反讽加上类型标签
for item in irony_data:
    item["type"] = "反讽"

# ==========================================
# 3. 合并与保存
# ==========================================
full_data = []
full_data.extend(gen_double_negation(100)) # 100条双重否定
full_data.extend(gen_concessive(50))       # 100条转折 (50对)
full_data.extend(gen_simple(50))           # 100条简单句 (50对)
full_data.extend(irony_data)               # 20条反讽 (虽然少，但很有代表性)

# 转换为 DataFrame
df_stress = pd.DataFrame(full_data)

# 打乱顺序
df_stress = df_stress.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存
df_stress.to_csv("logic_stress_test_large.csv", index=False)
print(f"✅ 已生成测试数据集，共 {len(df_stress)} 条。")
print(df_stress["type"].value_counts())