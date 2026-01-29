import os
import time
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 全局配置 ======================
MODEL_ID = 'doubao-seed-1-8-251228'
API_KEY = os.getenv('ARK_API_KEY')
BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3'
SAVE_PATH = 'data/processed/doubao_generated_650_samples.csv'
# 失败任务配置（反讽-影视娱乐调整为140条，适配模型输出）
FAILED_TASK_CONFIGS = [
    ("反讽-美食餐饮", "反讽", "美食餐饮", 80, "7:1", "6.9:1", "外卖餐品的评价、线下餐厅的打卡体验、对食材/口味/分量的吐槽/夸赞，贴合大众美食评价习惯"),
    ("反讽-影视娱乐", "反讽", "影视娱乐", 140, "7:1", "6.85:1", "追电视剧/看电影的评价、刷短视频的感受、对明星/综艺的轻松吐槽/夸赞，贴合年轻人的娱乐交流习惯"),
]

MAX_WORKERS = len(FAILED_TASK_CONFIGS)  # 多线程数适配任务数
# 反讽-影视娱乐专属条数容忍（贴合模型固定生成142条的实际情况）
IRONY_ENTERTAINMENT_TOLERANCE = (138, 142)
# ==========================
# =============================

# 1. 初始化OpenAI客户端
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

# 2. 定义枚举和Pydantic模型（与主代码一致，无修改）
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

class SentimentSample(BaseModel):
    text: str = Field(description='中文文本，6-35字，贴合真实场景，语法规范，语义自然', min_length=6, max_length=35)
    label: LabelEnum = Field(description='情感标签，0=负面，1=正面，与内容核心情感一致')
    type: SentenceTypeEnum = Field(description='句式类型，与内容核心语法特征一致')
    sub_type: SubTypeEnum = Field(description='子类型，严格遵循反讽子类型规则，非反讽为Normal')
    domain: DomainEnum = Field(description='内容领域，与内容主题高度贴合，无领域错位')
    is_valid: Literal[True] = Field(description='样本有效性，固定为True，所有样本均有效')

class SubTaskSampleList(BaseModel):
    samples: List[SentimentSample] = Field(description='子任务情感分析样本列表，符合所有核心规则和质量标准', min_length=1, max_length=200)

# 3. 全局System提示词（与主代码一致，无修改）
GLOBAL_SYSTEM_PROMPT = """
你是一名专业的中文自然语言处理数据生成专家，擅长生成贴合真实场景、语法规范、语义自然的中文情感分析样本，严格遵循以下**基础规则**和**质量标准**，所有生成内容必须同时满足：
### 【基础规则：字段含义与内容强关联】
所有生成样本包含6个字段：text、label、type、sub_type、domain、is_valid，字段含义与内容的关联规则为**硬约束**，必须100%遵守：
1. text：6-35字中文文本，内容必须与「domain领域」高度贴合，符合该领域的真实使用场景；
2. label：情感标签（0=负面/消极，1=正面/积极），标签必须与text的**核心情感**一致，无情感歧义；
3. type：句式类型，必须与text的**核心语法特征**一致（转折/双重否定/反讽）；
4. sub_type：子类型，仅3种取值，规则为：Normal（非反讽默认）、Negative_Ironic（负面反讽=label0）、Positive_Ironic（正面反讽=label1）；
5. domain：内容领域，围绕核心主题展开，无领域错位；
6. is_valid：固定值True，所有样本均有效。
### 【质量标准：4个必须】
1. 必须语法规范：无病句、无错别字、无冗余，符合中文日常表达；
2. 必须语义自然：贴合真实场景，无模板化表达；
3. 必须情感明确：label与核心情感一致，无歧义；
4. 必须句式纯正：type与核心语法特征一致，无错位。
### 【三大句式核心特征】
1. 转折：语义对立，可含转折连词或无连词但有转折语义，禁止普通并列/承接句；
2. 双重否定：两个否定词叠加表肯定，无歧义，否定词在同一语义单元；
3. 反讽：正话反说/反话正说，轻微吐槽/调侃，Negative_Ironic=label0、Positive_Ironic=label1（硬绑定）。
### 【计数与比例规则】
1. 比例为硬约束：label比例、反讽子类型比例偏差不超过±1；
2. 条数为软约束：优先保证质量和比例，最终条数精准即可；
### 【场景贴合标准】
内容真实、具体、生活化，贴合领域实际交流习惯，避免空洞抽象。
""".replace("\n", "").strip()

# 4. 子任务Prompt生成函数（与主代码一致，无修改）
def generate_subtask_prompt(task_name, sentence_type, domain, num, label_ratio, sub_type_ratio, domain_scene):
    sub_type_tip = f"子类型固定为Normal，无需生成反讽相关子类型" if sub_type_ratio == "Normal" else \
        f"子类型比例为Negative_Ironic:Positive_Ironic={sub_type_ratio}，严格遵守反讽子类型定义，比例偏差不超过±1"
    prompt = f"""
请生成{num}条符合全局基础规则的中文情感分析样本，本次子任务核心要求必须100%遵守：
1. 核心参数：句式类型={sentence_type}，领域={domain}，{num}条，label0:label1={label_ratio}，{sub_type_tip}；
2. 场景要求：严格贴合{domain}场景——{domain_scene}，具体生活化，无空洞抽象；
3. 句式要求：遵循{sentence_type}核心语法特征，自然变体，避免模板化；
4. 情感要求：label与核心情感一致，日常吐槽/夸赞，无极端情绪；
5. 内容要求：text6-35字，语法规范无错字，符合朋友圈/聊天框表达习惯；
6. 最终要求：内容质量＞场景贴合＞比例合规＞条数精准，允许微调，最终{num}条即可。
直接生成样本，无需额外解释、标注、格式化，样本独立成义无上下文关联。
""".replace("\n", "").strip()
    return prompt

# 5. 子任务执行函数（保留3次重试，反讽影视专属条数容忍，失败返回空列表）
def run_subtask(task_config, max_retries=3):
    task_name, sentence_type, domain, num, label_ratio, sub_type_ratio, domain_scene = task_config
    # 单独设置条数容忍范围：反讽-影视娱乐138~142，其他任务精准匹配
    if task_name == "反讽-影视娱乐":
        min_num, max_num = IRONY_ENTERTAINMENT_TOLERANCE
        num_tip = f"{min_num}~{max_num}条（容忍范围）"
    else:
        min_num = max_num = num
        num_tip = f"{num}条（精准要求）"

    for retry in range(1, max_retries+1):
        start_time = time.time()
        try:
            print(f"🔄 【{task_name}】重试{retry}/{max_retries}｜要求{num_tip}")
            # 生成prompt并调用模型
            prompt = generate_subtask_prompt(task_name, sentence_type, domain, num, label_ratio, sub_type_ratio, domain_scene)
            completion = client.beta.chat.completions.parse(
                model=MODEL_ID,
                messages=[{"role": "system", "content": GLOBAL_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                response_format=SubTaskSampleList,
                extra_body={"thinking": {"type": "enabled"}},
                temperature=1.2,
                max_tokens=16384,
                timeout=500,
            )
            # 解析结果并校验条数
            subtask_samples = completion.choices[0].message.parsed.samples
            actual_num = len(subtask_samples)
            if not (min_num <= actual_num <= max_num):
                raise Exception(f"条数超出范围，实际{actual_num}条")
            # 执行成功，打印日志并返回样本
            cost_time = round(time.time() - start_time, 2)
            print(f"✅ 【{task_name}】重试成功{retry}/{max_retries}｜实际{actual_num}条｜耗时{cost_time}秒")
            return subtask_samples
        except Exception as e:
            cost_time = round(time.time() - start_time, 2)
            print(f"❌ 【{task_name}】重试失败{retry}/{max_retries}｜错误：{str(e)}｜耗时{cost_time}秒")
            # 最后一次重试失败，打印最终提示并返回空列表
            if retry == max_retries:
                print(f"💥 【{task_name}】{max_retries}次重试全失败，跳过该任务")
    return []

# 6. 主重试逻辑（核心：单个任务成功即保存，失败任务跳过，无硬约束终止）
def retry_failed_tasks():
    # 初始化结果容器：按任务名存储成功样本，方便统计
    task_success_samples = {}
    total_expected_num = sum([task[3] for task in FAILED_TASK_CONFIGS])  # 总要求条数

    # 打印启动信息
    print(f"🚀 多线程重试启动｜线程数：{MAX_WORKERS}｜待重试任务：{len(FAILED_TASK_CONFIGS)}个｜总要求条数：{total_expected_num}")
    print(f"📌 反讽-影视娱乐专属配置：条数容忍138~142条，其他任务精准要求")
    print(f"{'='*80}\n")

    # 多线程执行所有失败任务
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务并映射：未来对象 → 任务名
        future2task = {executor.submit(run_subtask, cfg): cfg[0] for cfg in FAILED_TASK_CONFIGS}
        # 遍历已完成的任务，收集结果（成功存样本，失败存空）
        for future in as_completed(future2task):
            task_name = future2task[future]
            try:
                success_samples = future.result()
                task_success_samples[task_name] = success_samples
                if success_samples:
                    print(f"\n📦 【{task_name}】结果收集完成｜成功生成{len(success_samples)}条\n")
                else:
                    print(f"\n⚠️  【{task_name}】无有效样本｜跳过该任务\n")
            except Exception as e:
                task_success_samples[task_name] = []
                print(f"\n💥 【{task_name}】线程执行异常｜错误：{str(e)}｜跳过该任务\n")

    # 汇总所有成功任务的样本
    all_success_samples = []
    for task_name, samples in task_success_samples.items():
        all_success_samples.extend(samples)
    total_actual_num = len(all_success_samples)

    # 第一步：无任何成功样本，直接终止（无需操作CSV）
    if total_actual_num == 0:
        print(f"{'='*80}")
        print(f"❌ 所有任务均重试失败，无任何有效样本，终止保存")
        return

    # 打印详细的任务执行统计（核心：清晰看到每个任务的成功/失败状态）
    print(f"{'='*80}")
    print(f"📊 重试结果详细统计｜共{len(FAILED_TASK_CONFIGS)}个任务")
    for task_name in task_success_samples:
        sample_num = len(task_success_samples[task_name])
        status = "✅ 成功" if sample_num > 0 else "❌ 失败"
        print(f"  {status} 【{task_name}】｜生成样本：{sample_num}条")
    print(f"📊 整体统计｜总要求条数：{total_expected_num}｜实际成功条数：{total_actual_num}")
    if total_actual_num < total_expected_num:
        print(f"⚠️  部分任务失败，存在条数偏差（{total_expected_num - total_actual_num}条），仅保存有效数据")
    print(f"{'='*80}")

    # 第二步：读取原有CSV文件（文件不存在则终止，避免新建空文件）
    if not os.path.exists(SAVE_PATH):
        print(f"❌ 原有样本文件不存在：{SAVE_PATH}｜无法合并，终止保存")
        return
    df_original = pd.read_csv(SAVE_PATH, encoding='utf-8-sig')
    print(f"📁 读取原有CSV｜成功加载{len(df_original)}条历史样本")

    # 第三步：转换成功样本为DataFrame，合并到原有数据
    df_new = pd.DataFrame([s.model_dump() for s in all_success_samples])  # 新生成的成功样本
    df_merged = pd.concat([df_original, df_new], ignore_index=True)       # 合并历史+新样本

    # 第四步：覆盖保存到CSV（无论总条数是否达标，都保存有效数据）
    df_merged.to_csv(SAVE_PATH, index=False, encoding='utf-8-sig')

    # 打印最终完成信息，包含详细统计
    print(f"{'='*80}")
    print(f"✅ 重试任务执行完成｜有效数据已覆盖保存至：{SAVE_PATH}")
    print(f"📈 最终数据统计：")
    print(f"  - 历史样本条数：{len(df_original)}")
    print(f"  - 本次新增条数：{len(df_new)}")
    print(f"  - 合并后总条数：{len(df_merged)}")
    print(f"\n🔍 合并后样本质量分布：")
    print(f"  句式分布：\n{df_merged['type'].value_counts().to_string(index=True, indent=4)}")
    print(f"  反讽子类型分布：\n{df_merged[df_merged['type']=='反讽']['sub_type'].value_counts().to_string(index=True, indent=4)}")
    print(f"{'='*80}")

# 主入口
if __name__ == '__main__':
    try:
        retry_failed_tasks()
    except Exception as e:
        print(f"\n❌ 重试脚本全局异常｜错误信息：{str(e)}")