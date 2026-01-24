import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区 =================
# 建议使用环境变量，或者直接填入
API_KEY = os.getenv('DEEPSEEK_API_KEY')
BASE_URL = "https://api.deepseek.com"
# 推荐使用 R1 (reasoner) 保证逻辑质量
# 如果觉得慢，可以改用 "deepseek-chat"
MODEL_NAME = "deepseek-reasoner"

# 每种类型生成多少条？(建议 20-30 条，太长容易截断)
COUNT_PER_TYPE = 100
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 目标类型列表 =================
TARGET_TYPES = [
    "双重否定",
    "反讽",
    "转折关系",
    "简单句"
]

# ================= Prompt 模版 =================
def get_prompt(logic_type, count):
    return f"""
    请生成 {count} 条中文情感分析句子。
        【核心约束】
    1. **所有句子的逻辑类型必须是：[{logic_type}]**。
    2. 情感标签 (label) 只能是：'Positive' 或 'Negative'。
    3. 风格：口语化、真人叙述风格，而不是像AI生成的那样。
        【类型参考】
    - **双重否定**：我不得不承认这很好。(Positive)
    - **反讽**：这续航真是绝了，出门必带充电宝。(Negative)
    - **转折关系**：虽然很贵，但真香。(Positive)
    - **简单句**：这个电影真难看。(Negative)
        【输出格式】
    只返回一个 JSON 对象，包含 "items" 列表。
    **严禁**输出 markdown 标记 (```json)，**严禁**输出任何解释性文字。
        {{
        "items": [
            {{"text": "句子1...", "label": "Positive", "type": "{logic_type}"}},
            {{"text": "句子2...", "label": "Negative", "type": "{logic_type}"}}]
    }} 
    """

# ================= 主程序 =================
if __name__ == "__main__":
    print(f"[START] 开始生成数据 | 模型: {MODEL_NAME}")
    print(f"[PLAN] 计划生成: {len(TARGET_TYPES)} 个类型 x {COUNT_PER_TYPE} 条 = {len(TARGET_TYPES)*COUNT_PER_TYPE} 条")

    all_data = []
    # 外层循环：遍历类型
    for t_type in tqdm(TARGET_TYPES, desc="类别进度"):
        try:
            # 内层：单次 API 调用生成该类别的所有数据
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个严格输出JSON的数据构造专家。"},
                    {"role": "user", "content": get_prompt(t_type, COUNT_PER_TYPE)}
                ],
                response_format={'type': 'json_object'},
                temperature=1.3,  # 增加创造性
                max_tokens=8192  # 给足输出空间
            )

            content = response.choices[0].message.content

            # 解决可能的编码问题，确保文本格式无误
            content = content.replace("```json", "").replace("```", "").strip()

            # 处理 JSON 解析错误的可能性
            try:
                data_dict = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON 解析失败：{e}")
                continue

            items = data_dict.get("items", [])

            # 双重确认类型字段
            for item in items:
                item["type"] = t_type

            all_data.extend(items)
            print(f"  [OK] [{t_type}] 生成完成，获取 {len(items)} 条")

        except Exception as e:
            print(f"  [ERROR] [{t_type}] 生成失败: {e}")

    # ================= 保存 =================
    if all_data:
        df = pd.DataFrame(all_data)
        # 简单去重
        df.drop_duplicates(subset=["text"], inplace=True)

        filename = "deepseek_structured_data.csv"
        df.to_csv(filename, index=False)

        print("\n" + "="*40)
        print(f" 任务结束！共保存 {len(df)} 条数据。")
        print(f" 文件路径: {filename}")
        print("="*40)
        print(df["type"].value_counts())
    else:
        print(" 未生成任何数据。")
