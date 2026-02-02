import os
import time
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== ✅ 可调整参数配置区 =====================
# 1. 设置每次API请求生成的句子数量 (即 nums)
NUM_PER_TASK = 40  

# 2. 设置文件保存路径
SAVE_PATH = 'data/new/grid_generated_data_v1.csv'

# 3. 设置并发线程数 (建议 4-8)
MAX_WORKERS = 8 

# 其他固定配置
MODEL_ID = 'doubao-seed-1-8-251228'
API_KEY = os.getenv('ARK_API_KEY')
BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3'
# ==============================================================

# 1. 初始化OpenAI客户端
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# 2. 定义枚举
class LabelEnum(int, Enum):
    NEGATIVE = 0
    POSITIVE = 1

# ✅ 修改点1：添加 '简单句'
class SentenceTypeEnum(str, Enum):
    DOUBLE_NEGATIVE = '双重否定'
    IRONY = '反讽'
    TRANSITION = '转折'
    SIMPLE = '简单句'  # <--- 新增类型

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

# 3. 定义Pydantic模型
class SentimentSample(BaseModel):
    text: str = Field(min_length=6, max_length=35, description='中文文本')
    label: LabelEnum = Field(description='情感标签')
    type: SentenceTypeEnum = Field(description='句式类型')
    sub_type: SubTypeEnum = Field(description='子类型')
    domain: DomainEnum = Field(description='内容领域')
    is_valid: Literal[True] = Field(description='样本有效性')

class SubTaskSampleList(BaseModel):
    samples: List[SentimentSample] = Field(min_length=1, max_length=200)

# ===================== Prompt 定义 =====================
# ✅ 修改点2：在 System Prompt 中补充简单句的定义和约束
GLOBAL_SYSTEM_PROMPT = """
你是一名专业的中文自然语言处理数据生成专家，擅长生成贴合真实场景、语法规范、语义自然的中文情感分析样本，严格遵循以下**基础规则**和**质量标准**：

### 【基础规则】
1. text：6-35字中文文本，内容必须与「domain领域」高度贴合；
2. label：情感标签（0=负面，1=正面），必须与text核心情感一致；
3. type：句式类型，必须与text核心语法特征一致（转折/双重否定/反讽/简单句）；
4. sub_type：
   - Normal：非反讽句式（含简单句）默认值；
   - Negative_Ironic：负面反讽（正话反说，label=0）；
   - Positive_Ironic：正面反讽（反话正说，label=1）；
5. domain：内容领域，围绕该领域核心主题；
6. is_valid：固定为True。

### 【句式核心特征】
1. **转折**：语义对立，可含或不含关联词，逻辑必须转折。
2. **双重否定**：双重否定词叠加表达肯定，语气强烈。
3. **反讽**：正话反说或反话正说，带有吐槽/调侃/夸赞语气。
4. **简单句**：直陈语气，直接表达观点或陈述事实，结构清晰，逻辑单向。**禁止**包含转折连词、禁止双重否定、禁止阴阳怪气。

### 【质量标准】
语法规范、语义自然、情感明确、句式纯正。
""".replace("\n", "").strip()

def generate_subtask_prompt(task_name, sentence_type, domain, num, label_ratio, sub_type_ratio, domain_scene):
    if sub_type_ratio == "Normal":
        sub_type_tip = f"子类型固定为Normal"
    else:
        sub_type_tip = f"子类型比例为Negative_Ironic:Positive_Ironic={sub_type_ratio}"
    
    return f"""
请生成{num}条符合全局规则的样本：
1. 参数：句式={sentence_type}，领域={domain}，条数={num}，情感比例label0:label1={label_ratio}，{sub_type_tip}；
2. 场景：严格贴合{domain}的真实场景——{domain_scene}，具体生活化；
3. 要求：{sentence_type}特征明显，语义自然，无病句，text长度6-35字；
4. 输出：直接输出JSON数据，无需解释。
""".replace("\n", "").strip()

# ===================== 辅助配置：领域场景映射 =====================
DOMAIN_SCENES = {
    DomainEnum.SOCIAL: "朋友聊天、朋友圈感慨、同事吐槽、生活琐事分享",
    DomainEnum.LIFE_SERVICE: "快递外卖评价、家政维修体验、理发美容感受",
    DomainEnum.FOOD: "餐厅打卡评价、外卖口味吐槽、食材分量点评",
    DomainEnum.TRAVEL: "旅游景点打卡、酒店住宿体验、高铁飞机出行感受",
    DomainEnum.SHOPPING: "网购收货体验、商场逛街感受、商品质量/颜值评价",
    DomainEnum.ENTERTAINMENT: "电影电视剧评价、综艺追星吐槽、短视频刷片感受"
}

# ===================== 核心逻辑 =====================

def append_sample_to_csv(samples, save_path):
    """线程安全的追加保存"""
    try:
        df_new = pd.DataFrame([sample.model_dump() for sample in samples])
        header = not os.path.exists(save_path)
        df_new.to_csv(save_path, index=False, encoding='utf-8-sig', mode='a', header=header)
    except Exception as e:
        print(f"⚠️ 保存失败: {e}")

def run_single_api_task(sentence_type: SentenceTypeEnum, domain: DomainEnum, num: int):
    """
    单个API请求任务函数
    """
    start_time = time.time()
    
    # --- 自动配置参数逻辑 ---
    # 反讽句式特殊处理
    if sentence_type == SentenceTypeEnum.IRONY:
        label_ratio = "7:1" 
        sub_type_ratio = "6:1" 
    else:
        # 转折、双重否定、简单句 均使用平衡比例和Normal子类型
        label_ratio = "1:1"
        sub_type_ratio = "Normal"
    
    domain_scene = DOMAIN_SCENES.get(domain, f"{domain.value}的相关日常场景")
    task_name = f"{sentence_type.value}-{domain.value}"

    try:
        prompt = generate_subtask_prompt(
            task_name, sentence_type.value, domain.value, num, 
            label_ratio, sub_type_ratio, domain_scene
        )

        completion = client.beta.chat.completions.parse(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": GLOBAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=SubTaskSampleList,
            extra_body={"thinking": {"type": "enabled"}},
            temperature=1.2,
            max_tokens=8192,
            timeout=300,
        )
        
        result = completion.choices[0].message.parsed
        real_count = len(result.samples)
        
        if real_count > 0:
            append_sample_to_csv(result.samples, SAVE_PATH)

        cost_time = round(time.time() - start_time, 2)
        return {"status": "success", "task": task_name, "count": real_count, "time": cost_time}

    except Exception as e:
        cost_time = round(time.time() - start_time, 2)
        return {"status": "failed", "task": task_name, "error": str(e), "time": cost_time}

def generate_grid_data():
    """
    主控函数
    """
    if os.path.exists(SAVE_PATH):
        try:
            os.remove(SAVE_PATH)
            print(f"🗑️ 已清理旧文件: {SAVE_PATH}")
        except:
            pass

    # 自动遍历 Enum，现在包含简单句了
    all_tasks = []
    for s_type in SentenceTypeEnum:
        for dom in DomainEnum:
            all_tasks.append((s_type, dom))
    
    total_tasks = len(all_tasks)
    expected_total_samples = total_tasks * NUM_PER_TASK

    print(f"🚀 开始生成 | 任务数: {total_tasks} | 目标总数: {expected_total_samples} | 每次请求: {NUM_PER_TASK}条")
    print(f"⚡ 并发线程: {MAX_WORKERS} | 句式: {len(SentenceTypeEnum)}种 | 领域: {len(DomainEnum)}个")

    success_count = 0
    total_samples_generated = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(run_single_api_task, t[0], t[1], NUM_PER_TASK): f"{t[0].value}-{t[1].value}" 
            for t in all_tasks
        }
        
        with tqdm(total=total_tasks, desc="任务进度") as pbar:
            for future in as_completed(future_map):
                res = future.result()
                if res["status"] == "success":
                    success_count += 1
                    total_samples_generated += res["count"]
                    pbar.set_postfix_str(f"最新: {res['task']} ({res['count']}条)")
                else:
                    print(f"\n❌ {res['task']} 失败: {res['error']}")
                pbar.update(1)

    print(f"\n{'='*60}")
    print(f"🎉 所有任务结束")
    print(f"📊 成功任务: {success_count}/{total_tasks}")
    print(f"📝 实际生成数据: {total_samples_generated} 条")
    print(f"💾 文件保存路径: {SAVE_PATH}")
    
    if os.path.exists(SAVE_PATH):
        try:
            df = pd.read_csv(SAVE_PATH)
            print("\n🔍 数据分布预览:")
            print(df.groupby(['type', 'domain']).size().unstack(fill_value=0))
        except:
            pass

if __name__ == '__main__':
    generate_grid_data()