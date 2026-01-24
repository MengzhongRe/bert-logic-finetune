import pandas as pd
import random

# ==========================================
# 1. 定义语义库 (Semantic Knowledge Base)
# ==========================================
# 结构：{领域: {subjects: [], pos_adj: [], neg_adj: []}}
semantic_bank = {
    "food": {
        "subjects": ["这家餐厅", "这道牛排", "甜点", "烧烤", "外卖", "火锅底料", "海鲜", "自助餐"],
        "pos_adj": ["美味", "新鲜", "入味", "正宗", "丰盛", "干净", "嫩滑"],
        "neg_adj": ["难吃", "不新鲜", "太咸", "塞牙", "油腻", "有异味", "没熟"]
    },
    "tech": {
        "subjects": ["这款手机", "这个算法", "系统", "App", "网速", "电池续航", "屏幕显示", "人脸识别"],
        "pos_adj": ["流畅", "智能", "高效", "清晰", "耐用", "精准", "灵敏"],
        "neg_adj": ["卡顿", "弱智", "低效", "模糊", "发烫", "迟钝", "总是闪退"]
    },
    "media": {
        "subjects": ["这部电影", "剧情", "演员演技", "背景音乐", "这本小说", "结局", "特效"],
        "pos_adj": ["精彩", "感人", "震撼", "紧凑", "深刻", "逼真", "有创意"],
        "neg_adj": ["无聊", "狗血", "尴尬", "拖沓", "俗套", "五毛钱", "不知所云"]
    },
    "service": {
        "subjects": ["客服", "导游", "柜台服务", "售后", "快递小哥", "装修队", "保洁阿姨"],
        "pos_adj": ["负责", "热情", "细心", "专业", "准时", "耐心"],
        "neg_adj": ["敷衍", "粗鲁", "马虎", "业余", "拖延", "推卸责任"]
    }
}

# 辅助函数：从同一领域取词，保证语义连贯
def get_pair(sentiment="pos"):
    domain = random.choice(list(semantic_bank.keys()))
    sub = random.choice(semantic_bank[domain]["subjects"])
    if sentiment == "pos":
        adj = random.choice(semantic_bank[domain]["pos_adj"])
    else:
        adj = random.choice(semantic_bank[domain]["neg_adj"])
    return sub, adj

# ==========================================
# 2. 模版生成函数 (升级版)
# ==========================================

# A. 双重否定 (Target: Positive)
def gen_double_negation(n=100):
    data = []
    for _ in range(n):
        sub, adj = get_pair("pos") # 取正面词
        templates = [
            f"{sub}不是不{adj}。",
            f"{sub}不得不说是{adj}的。",
            f"对于{sub}，我没法不觉得{adj}。",
            f"我不否认{sub}很{adj}。",
            f"很难说{sub}是不{adj}的。"
        ]
        data.append({"text": random.choice(templates), "label": "Positive", "type": "双重否定"})
    return data

# B. 转折关系 (Target: Positive/Negative)
def gen_concessive(n=100):
    data = []
    # 这里需要特殊的“条件”词库，为了简单，我们用通用模版
    # 逻辑：虽然 [负面形容词A]，但是 [正面形容词B] -> Positive
    # 注意：A和B必须来自同一领域！
    for _ in range(n):
        domain = random.choice(list(semantic_bank.keys()))
        sub = random.choice(semantic_bank[domain]["subjects"])
        p_adj = random.choice(semantic_bank[domain]["pos_adj"])
        n_adj = random.choice(semantic_bank[domain]["neg_adj"])
        
        # 正面转折
        s1 = f"虽然{sub}有点{n_adj}，但整体还是挺{p_adj}的。"
        data.append({"text": s1, "label": "Positive", "type": "转折(正)"})
        
        # 负面转折
        s2 = f"虽然{sub}很{p_adj}，可是太{n_adj}了，无法接受。"
        data.append({"text": s2, "label": "Negative", "type": "转折(负)"})
    return data

# C. 简单句 (Baseline)
def gen_simple(n=100):
    data = []
    for _ in range(n):
        sub_p, adj_p = get_pair("pos")
        data.append({"text": f"{sub_p}很{adj_p}。", "label": "Positive", "type": "简单句"})
        
        sub_n, adj_n = get_pair("neg")
        data.append({"text": f"{sub_n}太{adj_n}了。", "label": "Negative", "type": "简单句"})
    return data

# D. 反讽句 (Irony) - 保持手动录入的高质量数据
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
    {"text": "这手机续航真强，充电两小时通话五分钟。", "label": "Negative"},
    {"text": "真是良心商家，涨价还减量。", "label": "Negative"},
    {"text": "这牛排真嫩，咬都咬不动。", "label": "Negative"},
    {"text": "这网速快得像蜗牛爬。", "label": "Negative"},
    {"text": "老板真大方，年终奖发了一箱空气。", "label": "Negative"}
]
for item in irony_data: item["type"] = "反讽"

# ==========================================
# 3. 执行生成
# ==========================================
full_data = []
full_data.extend(gen_double_negation(150))
full_data.extend(gen_concessive(100))
full_data.extend(gen_simple(100))
full_data.extend(irony_data * 5) # 复制几次反讽数据，增加权重

df_stress = pd.DataFrame(full_data)
df_stress = df_stress.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存
df_stress.to_csv("logic_stress_test_v2.csv", index=False)
print(f"✅ 生成完毕！共 {len(df_stress)} 条数据。")
print(df_stress['type'].value_counts())
print("示例：")
print(df_stress.head(5).to_markdown())