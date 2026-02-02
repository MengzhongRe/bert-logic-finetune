import os
import time
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= ✅ 全局配置区 =================
# 1. 保存路径 (建议使用 v2 区分)
SAVE_PATH = 'data/raw/data_aug.csv'

# 2. 单次请求最大生成条数 (核心修复：设为 20 以避免 Token 溢出)
MAX_BATCH_SIZE = 40  

# 3. 并发线程数
MAX_WORKERS = 8

# 4. 模型配置
MODEL_ID = 'doubao-seed-1-8-251228'
API_KEY = os.getenv('ARK_API_KEY')
BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3'
# =================================================

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ================= 1. 数据结构定义 =================
class LabelEnum(int, Enum):
    NEGATIVE = 0
    POSITIVE = 1

class SentenceTypeEnum(str, Enum):
    DOUBLE_NEGATIVE = '双重否定'
    IRONY = '反讽'
    TRANSITION = '转折'
    SIMPLE = '简单句'

class SubTypeEnum(str, Enum):
    NORMAL = 'Normal'
    NEG_IRNOY = 'Negative_Ironic'
    POS_IRONY = 'Positive_Ironic'

class DomainEnum(str, Enum):
    SOCIAL = "日常社交"
    LIFE_SERVICE = "生活服务"
    FOOD = "美食餐饮"
    TRAVEL = "出行旅游"
    SHOPPING = "购物消费"
    ENTERTAINMENT = "影视娱乐"

class SentimentSample(BaseModel):
    text: str = Field(min_length=6, max_length=35, description="生成的文本")
    label: LabelEnum
    type: SentenceTypeEnum
    sub_type: SubTypeEnum
    domain: DomainEnum
    is_valid: Literal[True]

class SubTaskSampleList(BaseModel):
    samples: List[SentimentSample] = Field(min_length=1, max_length=200)

# ================= 2. 核心 Prompt 生成逻辑 (深度优化版) =================
def generate_subtask_prompt(task_name, sentence_type, domain, num, label_ratio, sub_type_ratio, domain_scene):
    # 根据子类型定制思维链指令
    special_instruction = ""
    examples = ""

    # --- 针对【正面反讽】(Label 1, 反话正说) ---
    if sub_type_ratio == "Positive_Ironic":
        special_instruction = """
        【核心任务：生成“反话正说” (Positive Irony)】
        请生成**表面看似抱怨/责怪，实则表达极度喜爱/满意**的句子。
        - **逻辑公式**：表面负面词汇 (累/贵/烦/虐/恨) + 核心正面体验 = 极致的爱。
        - **常用套路**：
          1. "抱怨"东西太好导致了副作用（如：熬夜、长胖、没钱、不想走）。
          2. "责怪"商家/人做得太绝，不给别人留活路。
          3. 用"恨"来表达"爱"（如：恨死你了，干嘛这么懂我）。
        """
        examples = f"""
        【参考范例 (Positive Irony)】：
        - (影视) 这部剧简直有毒，害得我昨晚又熬到凌晨三点，黑眼圈都出来了！
        - (美食) 老板你心太黑了，做这么好吃是想让我这周减肥计划全泡汤吗？
        - (社交) 你这人真讨厌，每次送礼物都送到我心坎里，想拒绝都找不到理由。
        - (旅游) 这里的床舒服得太过分了，导致我错过了早上的日出，差评！
        """

    # --- 针对【负面反讽】(Label 0, 正话反说) ---
    elif sub_type_ratio == "Negative_Ironic":
        special_instruction = """
        【核心任务：生成“正话反说” (Negative Irony)】
        请生成**表面看似夸奖/客气，实则表达强烈不满/吐槽**的句子。
        - **逻辑公式**：表面褒义词 (聪明/贴心/大方/独特) + 荒谬/糟糕的事实 = 阴阳怪气的嘲讽。
        - **常用套路**：
          1. 夸奖对方的错误行为（如：你真行、由于你的“专业”）。
          2. 用高大上的词形容破烂的东西（如：复古=破，原生态=脏，热闹=拥挤）。
        """
        examples = f"""
        【参考范例 (Negative Irony)】：
        - (服务) 快递员真是“神行太保”，同城快递硬是走了五天还没到。
        - (旅游) 这酒店的隔音效果真“棒”，隔壁打呼噜听得一清二楚。
        - (社交) 您可真“大方”，聚会请客居然用优惠券抵扣了两块钱。
        - (美食) 这牛排煎得真有“嚼劲”，我腮帮子都嚼酸了还没烂。
        """

    # --- 针对【消极双重否定】(Label 0, 必须判负) ---
    elif sentence_type == "双重否定" and "1:0" in label_ratio:
        special_instruction = """
        【核心任务：生成“消极的双重否定”】
        请生成包含双重否定词（不得不、不能不、没法不），但**核心情感是负面/无奈/失望**的句子。
        让模型学会：双重否定不一定全是赞美，也可以是“不得不吐槽”。
        """
        examples = f"""
        【参考范例 (Negative Double Negative)】：
        - (旅游) 面对满地的垃圾，我不得不说这次旅行体验糟透了。
        - (服务) 这种敷衍的服务态度，没法不让人感到心寒。
        - (影视) 看到大结局强行煽情，观众不能不觉得这是在侮辱智商。
        """
    
    # --- 针对【困难转折】(Label 0/1) ---
    elif sentence_type == "转折":
        special_instruction = """
        【核心任务：生成“强干扰转折句”】
        前半句必须包含强烈的情感词（干扰项），但后半句通过转折完全反转情感。
        """
        examples = f"""
        【参考范例】：
        - (Label 1) 虽然房间小得连行李箱都打不开，但床品的舒适度真的没话说。
        - (Label 0) 这家店装修得富丽堂皇看着很高档，但菜品的味道简直是在欺诈。
        """

    else:
        special_instruction = "请生成符合指定句式特征和情感标签的句子，语法自然，贴合场景。"

    # 组装 Prompt
    prompt = f"""
你是一名资深的中文语言学家。当前任务：生成 **{num}条** 符合以下约束的样本。

### 🎯 任务参数
- **句式**：{sentence_type} | **领域**：{domain}
- **情感**：{label_ratio.replace('0:1', '【纯正面 Label=1】').replace('1:0', '【纯负面 Label=0】')}
- **场景**：{domain_scene}

### 💡 {special_instruction}

### 📚 {examples}

### ⚠️ 严格约束
1. **去模板化**：严禁所有句子都用同一个开头，变换主语。
2. **去标注**：输出纯文本，严禁包含 (反讽)、(褒义) 等括号解释。
3. **长度**：6-35字之间，口语化，接地气。

### 📤 输出格式
请直接返回包含 {num} 条样本的 JSON 数据。
""".strip()
    return prompt

# ================= 3. 任务列表 (基于评估结果的精准打击) =================
HARD_TASK_CONFIGS = [
    # Group A: 抢救“正面反讽” (召回率 0% - 20% 的重灾区) -> 240条
    ("正面反讽-影视娱乐", "反讽", "影视娱乐", 80, "0:1", "Positive_Ironic", "抱怨剧情太虐心、演员太会演害我哭、反向安利冷门剧"),
    ("正面反讽-生活服务", "反讽", "生活服务", 80, "0:1", "Positive_Ironic", "抱怨外卖送太快、抱怨美甲做太好洗碗不方便、抱怨私教太严格"),
    ("正面反讽-日常社交", "反讽", "日常社交", 80, "0:1", "Positive_Ironic", "凡尔赛式抱怨房子大、抱怨朋友送礼太贵重、抱怨聚会太嗨太累"),

    # Group B: 抢救“出行旅游” (全线崩盘区) -> 180条
    ("负面反讽-出行旅游", "反讽", "出行旅游", 100, "1:0", "Negative_Ironic", "夸景点人多热闹实则拥挤、夸酒店原生态实则破旧、夸导游精明实则坑人"),
    ("消极双重-出行旅游", "双重否定", "出行旅游", 80, "1:0", "Normal", "不得不吐槽行程安排、没法不让人觉得坑、不能不说是糟糕体验"),

    # Group C: 修复“转折误判” (生活/出行 Rec1 55%) -> 120条
    ("正面转折-出行旅游", "转折", "出行旅游", 60, "0:1", "Normal", "虽然路途遥远颠簸，但看到美景的那一刻值了"),
    ("正面转折-生活服务", "转折", "生活服务", 60, "0:1", "Normal", "虽然价格小贵，但这个服务态度和效果真的没话说"),

    # Group D: 巩固与平衡 (购物/美食/通用) -> 260条
    ("正面反讽-购物消费", "反讽", "购物消费", 100, "0:1", "Positive_Ironic", "吐槽东西太耐用想换都没理由、吐槽衣服质量好到穿不烂"),
    ("负面反讽-美食餐饮", "反讽", "美食餐饮", 80, "1:0", "Negative_Ironic", "夸分量精致实则吃不饱、夸味道清淡实则没味"),
    ("负面反讽-生活服务", "反讽", "生活服务", 80, "1:0", "Negative_Ironic", "夸维修师傅破坏力强、夸客服回复像机器人一样标准"),
]

# ================= 4. 执行逻辑 (自动拆单) =================
def run_subtask(config):
    task_name, s_type, domain, num, l_ratio, sub_ratio, scene = config
    
    # 扩大 max_tokens 以容纳思考过程，但因为拆单了，通常不会超
    current_max_tokens = 16384 
    
    try:
        prompt = generate_subtask_prompt(task_name, s_type, domain, num, l_ratio, sub_ratio, scene)
        completion = client.beta.chat.completions.parse(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            response_format=SubTaskSampleList,
            extra_body={"thinking": {"type": "enabled"}}, # 开启深度思考
            temperature=1.2,
            max_tokens=current_max_tokens,
            timeout=500, # 增加超时
        )
        samples = completion.choices[0].message.parsed.samples
        
        # 实时保存
        df = pd.DataFrame([s.model_dump() for s in samples])
        header = not os.path.exists(SAVE_PATH)
        # 使用 mode='a' 追加写入
        df.to_csv(SAVE_PATH, mode='a', header=header, index=False, encoding='utf-8-sig')
        
        return f"✅ {task_name}: 成功生成 {len(samples)} 条"
    except Exception as e:
        error_msg = str(e)
        if "length limit" in error_msg:
            return f"❌ {task_name}: Token溢出 (请进一步减小 MAX_BATCH_SIZE)"
        return f"❌ {task_name}: 失败 - {error_msg[:100]}..."

def generate_targeted_data():
    # 1. 清理旧文件
    if os.path.exists(SAVE_PATH):
        try:
            os.remove(SAVE_PATH)
            print(f"🗑️ 已删除旧文件: {SAVE_PATH}")
        except: pass

    print(f"🚀 启动增强生成 | 目标: ~800条 | 策略: 自动拆单 (每批次最大 {MAX_BATCH_SIZE} 条)")
    
    # 2. 自动拆分大任务
    split_tasks = []
    for config in HARD_TASK_CONFIGS:
        t_name, s_type, dom, total_num, l_ratio, s_ratio, scene = config
        
        if total_num <= MAX_BATCH_SIZE:
            split_tasks.append(config)
        else:
            # 拆分逻辑
            left = total_num
            part = 1
            while left > 0:
                current_batch = min(left, MAX_BATCH_SIZE)
                new_name = f"{t_name}_Part{part}"
                split_tasks.append((new_name, s_type, dom, current_batch, l_ratio, s_ratio, scene))
                left -= current_batch
                part += 1
    
    print(f"📊 原始任务: {len(HARD_TASK_CONFIGS)} -> 拆分后任务: {len(split_tasks)} (避免Token溢出)")

    # 3. 并行执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_subtask, cfg) for cfg in split_tasks]
        for future in tqdm(as_completed(futures), total=len(futures)):
            print(future.result())
            
    # 4. 最终统计
    if os.path.exists(SAVE_PATH):
        try:
            df = pd.read_csv(SAVE_PATH)
            print(f"\n🎉 增强生成完成！共 {len(df)} 条。")
            print("数据分布概览:")
            print(df.groupby(['type', 'label']).size())
        except: pass

if __name__ == '__main__':
    generate_targeted_data()