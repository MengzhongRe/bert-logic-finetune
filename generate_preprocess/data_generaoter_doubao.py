import os
import time
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 全局配置项 =====================
MODEL_ID = 'doubao-seed-1-8-251228'
API_KEY = os.getenv('ARK_API_KEY')
BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3'
SAVE_PATH = 'data/processed/doubao_generated_650_samples.csv'
MAX_WORKERS = 4  # 建议4-5，平衡并行度和生成质量
# ======================================================

# 1. 初始化OpenAI客户端（全局单例）
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

# 2. 定义枚举（与原代码一致，字段值与新Prompt规则完全匹配）
class LabelEnum(int, Enum):
    NEGATIVE = 0
    POSITIVE = 1

class SentenceTypeEnum(str, Enum):
    DOUBLE_NEGATIVE = '双重否定'
    IRONY = '反讽'
    TRANSITION = '转折'

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

# 3. 定义Pydantic模型（与原代码一致，保留长度/类型约束）
class SentimentSample(BaseModel):
    text: str = Field(
        description='中文文本，6-35字，贴合真实场景，语法规范，语义自然',
        min_length=6,
        max_length=35,
    )
    label: LabelEnum = Field(description='情感标签，0=负面，1=正面，与内容核心情感一致')
    type: SentenceTypeEnum = Field(description='句式类型，与内容核心语法特征一致')
    sub_type: SubTypeEnum = Field(description='子类型，严格遵循反讽子类型规则，非反讽为Normal')
    domain: DomainEnum = Field(description='内容领域，与内容主题高度贴合，无领域错位')
    is_valid: Literal[True] = Field(description='样本有效性，固定为True，所有样本均有效')

class SubTaskSampleList(BaseModel):
    samples: List[SentimentSample] = Field(
        description='子任务情感分析样本列表，符合所有核心规则和质量标准',
        min_length=1,
        max_length=200,
    )

# ===================== 新Prompt体系核心 =====================
# 全局System提示词：定义基础规则+字段含义+句式特征+质量标准
GLOBAL_SYSTEM_PROMPT = """
你是一名专业的中文自然语言处理数据生成专家，擅长生成贴合真实场景、语法规范、语义自然的中文情感分析样本，严格遵循以下**基础规则**和**质量标准**，所有生成内容必须同时满足：

### 【基础规则：字段含义与内容强关联】
所有生成样本包含6个字段：text、label、type、sub_type、domain、is_valid，字段含义与内容的关联规则为**硬约束**，必须100%遵守：
1. text：6-35字中文文本，内容必须与「domain领域」高度贴合，符合该领域的真实使用场景；
2. label：情感标签（0=负面/消极，1=正面/积极），标签必须与text的**核心情感**一致，无情感歧义；
3. type：句式类型，必须与text的**核心语法特征**一致（转折/双重否定/反讽）；
4. sub_type：子类型，仅3种取值，规则为：
   - Normal：非反讽句式（转折/双重否定）的默认子类型，无额外约束；
   - Negative_Ironic：负面反讽，核心情感为负面，通过**正话反说**表达（如“这道菜太好吃了，咸到我怀疑人生”）；
   - Positive_Ironic：正面反讽，核心情感为正面，通过**反话正说**表达（如“你可真行，居然能找到这么好吃的小众餐厅”）；
5. domain：内容领域，内容必须围绕该领域的**核心主题**展开，无领域错位；
6. is_valid：固定值True，所有样本均为有效样本。

### 【质量标准：4个必须】
1. 必须**语法规范**：无病句、无错别字、无冗余成分，符合中文日常表达习惯；
2. 必须**语义自然**：贴合真实使用场景，无生硬模板化表达，避免机械套用固定句式；
3. 必须**情感明确**：label与text核心情感高度一致，无“情感模糊/标签错位”；
4. 必须**句式纯正**：type与text的核心语法特征一致，无“句式特征不明显/句式错位”。

### 【三大句式核心特征（核心，必须严格遵守）】
#### 1. 转折句式（type=转折）
- 核心特征：**语义对立**（前半句与后半句表达的语义/情感相反/对立），存在明显的“转折逻辑”；
- 语法特征：可含转折连词（虽然/但是/尽管/却/然而/看似/实则/不过/只是），也可**无转折连词但存在转折语义**；
- 禁止：无转折语义的普通并列句/承接句。

#### 2. 双重否定句式（type=双重否定）
- 核心特征：**通过两个否定词叠加表达肯定语义**，语气比直接肯定更强烈，无否定歧义；
- 语法特征：含常见否定词（不/无/非/没/未/莫），两个否定词需在**同一个语义单元**中；
- 禁止：两个否定词表达否定语义、否定词不在同一语义单元。

#### 3. 反讽句式（type=反讽）
- 核心特征：**正话反说/反话正说**，通过与核心情感相反的表述表达真实情感，带有轻微的吐槽/调侃/夸赞语气，是**日常交流中的自然反讽**，非书面化讽刺；
- 子类型与情感的硬约束：
  - Negative_Ironic（负面反讽）：真实情感为负面（label=0），通过**正话反说**表达；
  - Positive_Ironic（正面反讽）：真实情感为正面（label=1），通过**反话正说**表达；
- 禁止：无反讽特征的普通感叹句/吐槽句。

### 【计数与比例规则：软约束计数，硬约束比例】
1. 比例为**硬约束**：如要求Negative_Ironic:Positive_Ironic=6:1，则生成的反讽样本中，负面反讽数量必须约为正面反讽的6倍，比例偏差不超过±1；
2. 条数为**软约束**：最终生成指定条数即可，无需逐句计数，优先保证内容质量和比例合规，允许生成过程中略有调整，最终条数精准即可；
3. label比例为**硬约束**：如要求label=0:1=7:1，则该批次样本中，负面情感数量必须约为正面情感的7倍，比例偏差不超过±1。

### 【场景贴合标准：真实、具体、生活化】
所有生成内容必须贴合「domain领域」的**真实日常使用场景**，避免空洞、抽象的表达，内容要具体，符合该领域的实际交流习惯。
""".replace("\n", "").strip()

# 重构后的子任务配置：场景具象化+规则精细化，总条数650
TASK_CONFIGS = [
    ("双重否定-日常社交", "双重否定", "日常社交", 60, "1:1", "Normal", "朋友线下/线上聊天、朋友圈日常感慨、和同事的轻松吐槽，内容围绕生活小事、心情、日常经历"),
    ("反讽-日常社交", "反讽", "日常社交", 80, "7:1", "6.9:1", "朋友之间的轻松互侃、朋友圈对生活小事的吐槽、和熟人的日常调侃，语气轻松，无恶意"),
    ("反讽-生活服务", "反讽", "生活服务", 70, "6:1", "6:1", "家政保洁/家电维修的体验吐槽、快递/外卖配送的评价、理发店/奶茶店的线下服务体验"),
    ("反讽-美食餐饮", "反讽", "美食餐饮", 80, "7:1", "6.9:1", "外卖餐品的评价、线下餐厅的打卡体验、对食材/口味/分量的吐槽/夸赞，贴合大众美食评价习惯"),
    ("反讽-出行旅游", "反讽", "出行旅游", 70, "6:1", "6:1", "景区游玩的体验、高铁/飞机/打车的出行感受、酒店/民宿的住宿体验，围绕旅游中的真实经历"),
    ("转折-购物消费", "转折", "购物消费", 30, "1:1", "Normal", "网购商品的收货体验、线下商场逛街的购物感受、日常用品的使用体验，围绕商品质量/颜值/实用性"),
    ("转折-出行旅游", "转折", "出行旅游", 50, "1:1", "Normal", "景区游玩的体验对比、交通出行的感受变化、酒店住宿的优缺点对比，贴合旅游中的真实体验"),
    ("转折-生活服务", "转折", "生活服务", 50, "1:1", "Normal", "家政服务的优缺点、快递外卖的速度与质量对比、线下门店服务的体验变化，贴合日常服务场景"),
    ("反讽-影视娱乐", "反讽", "影视娱乐", 140, "7:1", "6.85:1", "追电视剧/看电影的评价、刷短视频的感受、对明星/综艺的轻松吐槽/夸赞，贴合年轻人的娱乐交流习惯"),
]

# 重构后的子任务Prompt生成函数
def generate_subtask_prompt(task_name, sentence_type, domain, num, label_ratio, sub_type_ratio, domain_scene):
    if sub_type_ratio == "Normal":
        sub_type_tip = f"子类型固定为Normal，无需生成反讽相关子类型"
    else:
        sub_type_tip = f"子类型比例为Negative_Ironic:Positive_Ironic={sub_type_ratio}，严格遵守全局规则中反讽子类型的定义，比例偏差不超过±1"
    
    prompt = f"""
请你生成{num}条符合**全局基础规则**的中文情感分析样本，本次子任务的核心要求如下，必须100%遵守：
1. 核心参数：句式类型={sentence_type}，内容领域={domain}，生成总条数={num}条，情感标签比例label0(负面):label1(正面)={label_ratio}，{sub_type_tip}；
2. 场景要求：内容必须严格贴合{domain}领域的真实使用场景——{domain_scene}，内容要具体、生活化，符合该场景的日常交流习惯，避免空洞抽象；
3. 句式要求：严格遵守全局规则中{sentence_type}句式的**核心语法特征和质量标准**，允许使用符合特征的自然变体句式，优先保证内容的自然度和语法规范，避免机械模板化表达；
4. 情感要求：label标签必须与内容的核心情感高度一致，无情感歧义，负面情感为日常吐槽/不满，正面情感为日常夸赞/开心，均为生活化的普通情感，无极端情绪；
5. 内容要求：text长度为6-35字，语法规范、无病句、无错别字、无冗余成分，符合中文日常口头/书面交流习惯（如朋友圈、聊天框的表达）；
6. 最终要求：优先保证**内容质量＞场景贴合度＞比例合规＞条数精准**，允许生成过程中对条数略有调整，最终生成{num}条精准样本即可，无需逐句计数。

请直接生成符合要求的样本，无需额外解释、无需标注、无需格式化输出，所有样本均需独立成义，无上下文关联。
""".replace("\n", "").strip()
    return prompt
# ============================================================

# 成功任务样本追加保存函数（保留原逻辑，稳定可靠）
def append_sample_to_csv(samples, save_path):
    df_new = pd.DataFrame([sample.model_dump() for sample in samples])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        df_new.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"📁 新建CSV文件，保存{len(df_new)}条样本至：{save_path}")
    else:
        df_new.to_csv(save_path, index=False, encoding='utf-8-sig', mode='a', header=False)
        print(f"📁 追加{len(df_new)}条样本至：{save_path}")

# 子任务执行函数（保留原逻辑，替换Prompt调用）
def run_subtask(task_config):
    task_name, sentence_type, domain, num, label_ratio, sub_type_ratio, domain_scene = task_config
    start_time = time.time()
    try:
        # 生成子任务Prompt
        prompt = generate_subtask_prompt(task_name, sentence_type, domain, num, label_ratio, sub_type_ratio, domain_scene)
        # 调用API：使用全局System提示词+子任务Prompt
        completion = client.beta.chat.completions.parse(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": GLOBAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=SubTaskSampleList,
            extra_body={"thinking": {"type": "enabled"}},  # 保留深度思考，提升规则遵守度
            temperature=1.2,  # 微调温度：1.2兼顾多样性和稳定性（原1.3略高，易出现规则偏差）
            max_tokens=16384,
            timeout=400,  # 微调超时：400秒≈6.5分钟，足够子任务生
        )
        subtask_result = completion.choices[0].message.parsed
        # 条数校验
        if len(subtask_result.samples) != num:
            raise Exception(f"生成条数不符，要求{num}条，实际{len(subtask_result.samples)}条")
        # 实时追加保存成功样本
        append_sample_to_csv(subtask_result.samples, SAVE_PATH)
        # 计算耗时
        cost_time = round(time.time() - start_time, 2)
        return {
            "task_name": task_name,
            "samples": subtask_result.samples,
            "status": "success",
            "cost_time": cost_time,
            "num": num
        }
    except Exception as e:
        cost_time = round(time.time() - start_time, 2)
        return {
            "task_name": task_name,
            "samples": [],
            "status": "failed",
            "error": str(e),
            "cost_time": cost_time,
            "num": num
        }

# 核心主函数：保留原逻辑，进度可视化+实时保存+异常处理
def generate_samples_with_multithread():
    total_start = time.time()
    all_samples = []
    success_tasks = 0
    failed_tasks = []

    # 提前删除旧文件，避免样本混合
    if os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)
        print(f"🗑️  删除旧样本文件：{SAVE_PATH}")

    # 打印启动信息
    print(f"🚀 启动多线程样本生成｜总任务数：{len(TASK_CONFIGS)}｜并发线程数：{MAX_WORKERS}｜目标总条数：650")
    print(f"📌 模型：{MODEL_ID}｜核心原则：约束性+多样性+自然度+场景化\n")

    # 线程池调度
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(run_subtask, config): config[0] for config in TASK_CONFIGS}
        for future in tqdm(as_completed(future_to_task), total=len(TASK_CONFIGS), desc="整体任务进度"):
            task_name = future_to_task[future]
            try:
                task_result = future.result()
                if task_result["status"] == "success":
                    success_tasks += 1
                    all_samples.extend(task_result["samples"])
                    print(f"✅ 任务完成｜{task_name}｜{task_result['num']}条｜耗时{task_result['cost_time']}秒")
                else:
                    failed_tasks.append(task_result)
                    print(f"❌ 任务失败｜{task_name}｜错误：{task_result['error']}")
            except Exception as e:
                print(f"❌ 任务异常｜{task_name}｜未知错误：{str(e)}")
                failed_tasks.append({"task_name": task_name, "error": str(e)})

    # 全局结果统计
    total_cost = round(time.time() - total_start, 2)
    actual_total = len(all_samples)
    print(f"\n{'='*70}")
    print(f"📊 多线程生成任务统计｜总耗时：{total_cost}秒")
    print(f"✅ 成功任务数：{success_tasks}｜❌ 失败任务数：{len(failed_tasks)}")
    print(f"📈 实际生成总条数：{actual_total}｜🔖 目标总条数：650")
    if failed_tasks:
        print(f"⚠️  失败任务：{[t['task_name'] for t in failed_tasks]}")
        print(f"💡 解决方案：运行重试脚本补全失败任务即可，已成功样本已保存至CSV！")
    print(f"✅ 已生成的样本已实时保存至：{SAVE_PATH}")
    print(f"{'='*70}")

    # 全部成功时打印详细分布
    if actual_total == 650:
        df = pd.read_csv(SAVE_PATH, encoding='utf-8-sig')
        print(f"\n🔍 样本质量统计（650条完整）：")
        print(f"1. 句式分布：\n{df['type'].value_counts()}")
        print(f"\n2. 领域分布：\n{df['domain'].value_counts()}")
        print(f"\n3. 情感标签分布：\n{df['label'].value_counts()}")
        print(f"\n4. 反讽子类型分布：\n{df[df['type'] == '反讽']['sub_type'].value_counts()}")
        print(f"\n5. 前5条样本预览（自然度/场景化校验）：")
        print(df[['text', 'type', 'domain', 'label']].head(5).to_string(index=False))

    return actual_total

# 主程序入口
if __name__ == '__main__':
    try:
        generate_samples_with_multithread()
    except Exception as e:
        print(f"\n❌ 全局任务异常，错误信息：{str(e)}")
        print(f"💡 关键提示：已成功的任务样本已实时保存至{SAVE_PATH}，无需重新运行全部！")