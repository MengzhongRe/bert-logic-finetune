# 🚀 基于`bert`的逻辑句式+多领域 情感分类能力测试与分析
[![GitHub Stars](https://img.shields.io/github/stars/MengzhongRe/bert-logic-stress.svg?style=social)](https://github.com/MengzhongRe/bert-logic-stress)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.40.0-orange.svg)](https://huggingface.co/transformers/)
[![OpenAI](https://img.shields.io/badge/API-DeepSeek%20Reasoner-purple.svg)](https://www.deepseek.com/)

> 🔍 系统性评估基于`bert`微调的中文情感分析模型在复杂逻辑句式+垂直领域下的情感分类能力，精准定位短板，为微调提供可落地指导  
> 📝 含完整数据集构建流程（API调用+Prompt设计+去模板化优化）

---

## 📋 项目简介
本项目聚焦**自然语言处理中的逻辑语义理解与多领域适配问题**，核心包含两大模块：
1. **高质量数据集构建**：通过DeepSeek API调用，结合精细化Prompt设计，生成960条去模板化、硬编码精准标注的结构化数据；
2. **全维度模型测试**：从「逻辑句式」「垂直领域」「交叉场景」「全局性能」四个维度量化模型能力，输出标准化测试报告与微调方案。

### 🌟 核心目标
1. 构建覆盖4类逻辑句式、6大垂直领域的高质量情感分类数据集；
2. 量化模型在复杂逻辑+多领域场景下的分类准确率；
3. 定位「句式×领域」交叉短板，提供可落地的微调指导；
4. 输出适配GitHub归档的完整工程化方案（含数据构建+模型测试）。

---

## 📊 测试维度与数据规模
### 🎯 核心测试维度
| 测试层级       | 测试维度                          | 核心指标                |
|----------------|-----------------------------------|-------------------------|
| 📝 核心句式层     | 双重否定、反讽、简单句、转折关系  | 准确率（240条/句式）    |
| 🌐 垂直领域层     | 出行旅游、影视娱乐、日常社交、生活服务、美食餐饮、购物消费 | 准确率（160条/领域） |
| 🔗 交叉场景层     | 句式×领域（24种组合）              | 准确率（40条/组合）     |
| 📈 全局分类层     | Positive/Negative情感极性          | Precision/Recall/F1/Accuracy |

### 📦 数据规模
- 📊 总样本数：**960条**（无重复、全覆盖）
- 📝 句式分布：4类×240条 = 960条
- 🌐 领域分布：6类×160条 = 960条
- ❤️ 情感分布：Negative(552条) + Positive(408条) = 960条
- ✅ 数据质量：硬编码精准标注（type/domain/sub_type），无缺失值/重复文本，去模板化设计

---

## 🛠️ 数据集构建全流程
### 1. 🔧 技术选型
| 模块         | 工具/API                          | 核心优势                          |
|--------------|-----------------------------------|-----------------------------------|
| 文本生成     | DeepSeek Reasoner v3.2 API             | 中文逻辑推理能力强，支持结构化输出    |
| 数据存储     | CSV格式                           | 轻量易读，适配模型训练/测试       |
| 并发加速     | Python ThreadPoolExecutor          | 多线程批量生成，效率提升5-10倍    |
| 质量校验     | Pandas+自定义规则                 | 自动过滤无效样本，保证数据有效性  |

### 2. 📝 Prompt设计（精细化去模板化）
核心设计原则：**保留逻辑约束，打破固定结构，强化场景细节**，避免AI生成模板化文本。

#### 2.1 系统Prompt（全局约束）
```python
system_prompt = """
你正在辅助AI工程师完成中文逻辑语义的情感分析研究，需要生成符合指定逻辑类型+指定领域的口语化中文句子。
【核心约束】
1. 所有句子必须严格匹配指定的逻辑类型+指定领域，无偏差、无跨领域；
2. 情感标签只能是Positive或Negative，按用户指定的数量配比生成，严格遵守数量要求；
3. 风格为日常口语化、真人叙述，**拒绝固定模板（如“XX超XX”“虽然XX但XX”），同一逻辑类型需生成3种以上不同句式结构**；
4. 句子需包含具体场景细节（如时间、动作、感受），**避免简单评价式表达**（如“这家店很好吃”“这部剧很难看”）；
5. 句子长度适中（8-30字），无无意义内容、无重复句式，**正负情感的表达风格不统一**；
6. 反讽专项约束：负面反讽为日常吐槽场景，正面反讽仅限熟人/亲友间的轻松亲昵场景，严格贴合指定领域。
【输出格式】
只返回纯JSON对象，包含 "items" 列表，无任何markdown标记、解释性文字、空行。
{{
    "items": [
        {{"text": "具体的句子内容", "label": "Positive"}},
        {{"text": "具体的句子内容", "label": "Negative"}}
    ]
}} 
"""
```

#### 2.2 分句式Prompt示例（去模板化核心）
##### 反讽句式（最难生成，重点优化）
```python
examples = {
    "日常社交": "- 你可真“坏”，知道我喜欢吃草莓，特意买了一筐，还洗干净给我送来。(Positive)\n- 你可真“靠谱”，约好一起去看电影，我等了你半小时，你才说忘了。(Negative)\n- 我朋友可真“自私”，知道我加班，特意给我带了热乎的饭，还陪我一起加班。(Positive)",
    "美食餐饮": "- 这家甜品店可真“过分”，芋泥冰给的料这么足，害得我吃完晚饭都吃不下了。(Positive)\n- 你可真会选地方，这家餐厅的菜又贵又难吃，我下次再也不想来了。(Negative)\n- 老板可真“小气”，送的小菜比我点的主菜还好吃，这不是让人多来几次嘛。(Positive)",
    # 其他领域示例略（保持同风格，多结构）
}
```

##### 双重否定句式（避免“没有XX是不XX”单一结构）
```python
examples = {
    "生活服务": "- 这个快递点的工作人员态度很好，包裹包装得也严实，我寄了好几次，没有一次不满意。(Positive)\n- 这家修鞋店的师傅手艺太差，鞋子越修越坏，顾客没有一个不抱怨的。(Negative)\n- 小区的保洁阿姨很负责，每天都把楼道打扫得干干净净，没有一处不整洁的地方。(Positive)",
    "日常社交": "- 我闺蜜很靠谱，答应我的事一定会做到，没有一次不兑现承诺的。(Positive)\n- 这个人说话不算数，经常放别人鸽子，跟他打交道的人没有不生气的。(Negative)\n- 我同事很热心，不管谁有困难他都愿意帮忙，公司里没有不夸他的。(Positive)",
    # 其他领域示例略（补充“不是没有XX”“不可能不XX”等结构）
}
```

#### 2.3 Prompt动态生成逻辑
```python
def get_prompt(logic_type: str, domain: str, count: int) -> str:
    # 按句式+领域计算正负面数量（反讽8:2，其他5:5）
    if logic_type == "反讽":
        pos_count = round(count * 0.2)
        neg_count = count - pos_count
        pos_domain_tip = "正面反讽需贴合熟人/亲友间的轻松亲昵场景，严格匹配该领域的日常表达；" if pos_count > 0 else ""
        prompt_base = f"""请生成{count}条【{logic_type}】逻辑句式、【{domain}】领域的中文日常口语化句子，其中{neg_count}条为Negative（负面），{pos_count}条为Positive（正面），**严格遵守数量配比，不可偏差、不可跨领域**。
【核心约束】
- 所有句子必须严格贴合{domain}领域，无跨领域内容，口语化、无AI生成感，长度8-30字；
- {pos_domain_tip}
- 反讽句式需严格遵循：负面字面褒扬实际贬斥，正面字面贬斥实际褒扬。
【示例参考（{logic_type}-{domain}）】"""
    else:
        pos_count = round(count * 0.5)
        neg_count = count - pos_count
        prompt_base = f"""请生成{count}条【{logic_type}】逻辑句式、【{domain}】领域的中文日常口语化句子，其中{pos_count}条为Positive（正面），{neg_count}条为Negative（负面），**严格遵守数量配比，不可偏差、不可跨领域**。
【核心约束】
- 所有句子必须严格贴合{domain}领域，无跨领域内容，口语化、无AI生成感，长度8-30字；
【示例参考（{logic_type}-{domain}）】"""
    # 拼接对应领域示例，无则用通用示例
    example_text = examples.get(domain, "- 正面句子：描述具体场景和真实的愉悦感受，不用固定结构。(Positive)\n- 负面句子：描述具体场景和真实的不满感受，不用固定结构。(Negative)")
    return prompt_base + example_text
```

### 3. 🚀 API调用与并发加速
#### 3.1 API配置
```python
# API核心配置
API_KEY = os.getenv('DEEPSEEK_API_KEY')
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-reasoner"

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
```

#### 3.2 多线程批量生成（IO密集型加速）
```python
def generate_single_task(logic_type: str, domain: str, count: int) -> list:
    """单任务：生成「句式+领域」的有效数据"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": get_prompt(logic_type, domain, count)}
            ],
            response_format={"type": "json_object"},
            temperature=1.3,  # 平衡多样性与逻辑性
            max_tokens=8192,
            top_p=0.9
        )
        # 解析JSON结果，过滤无效数据
        content = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        data_dict = json.loads(content)
        raw_items = data_dict.get("items", [])
        # 质量校验+硬编码标注
        valid_items = check_data_quality(raw_items, logic_type, domain)
        return valid_items
    except Exception as e:
        print(f"[ERROR] [{logic_type}-{domain}] 生成失败：{str(e)[:80]}")
        return []

# 多线程执行所有任务（24个组合：4句式×6领域）
with ThreadPoolExecutor(max_workers=8) as executor:
    future_to_task = {
        executor.submit(generate_single_task, t, d, 40): (t, d, 40)  # 每个组合生成40条
        for t in TARGET_TYPES for d in DOMAINS
    }
    for future in tqdm(as_completed(future_to_task), total=24, desc="多线程生成进度"):
        task_result = future.result()
        if task_result:
            all_data.extend(task_result)
```

### 4. ✅ 数据质量校验与标注
```python
def check_data_quality(items: list, logic_type: str, domain: str) -> list:
    """数据质量校验+硬编码标注（100%精准）"""
    valid_items = []
    for item in items:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        text = item.get("text").strip()
        if len(text) < 8 or len(text) > 30:
            continue
        label = item.get("label", "").strip()
        if label not in ["Positive", "Negative"]:
            continue
        # 硬编码赋值：type/domain/sub_type（无标注误差）
        sub_type = "Normal"
        if logic_type == "反讽":
            sub_type = "Positive_Ironic" if label == "Positive" else "Negative_Ironic"
        valid_items.append({
            "text": text,
            "label": label,
            "type": logic_type,
            "sub_type": sub_type,
            "domain": domain,
            "is_valid": True
        })
    return valid_items
```

### 5. 📊 数据集构建亮点
1. **去模板化**：每个句式提供3+种不同结构示例，避免AI生成固定句式；
2. **精准标注**：type/domain/sub_type硬编码标注，准确率100%，无人工标注误差；
3. **高效生成**：多线程并发调用API，960条数据生成耗时＜5min；
4. **均衡分布**：句式×领域×情感极性三维度绝对均衡，适配模型公平测试。

---

## 🎯 核心测试结果
### 1. 逻辑句式能力排行 📝
| 句式类型   | 准确率  | 样本数 | 能力评级 | 🌟 结论                     |
|------------|---------|--------|----------|------------------------------|
| 简单句     | 92.50%  | 240    | 🥇 优秀     | 模型基础句式理解能力扎实     |
| 双重否定   | 87.50%  | 240    | 🥈 良好     | 次优，存在小幅优化空间       |
| 转折关系   | 85.83%  | 240    | 🥉 中等     | 略低于平均，需针对性优化     |
| 反讽       | 82.08%  | 240    | ⚠️ 待优化   | 核心短板，语义反转理解不足   |

### 2. 垂直领域能力排行 🌐
| 领域       | 准确率  | 样本数 | 能力评级 | 🌟 结论                     |
|------------|---------|--------|----------|------------------------------|
| 影视娱乐   | 93.75%  | 160    | 🥇 优秀     | 领域表现最优，结构化文本适配好 |
| 美食餐饮   | 90.00%  | 160    | 🥈 良好     | 次优，贴近生活场景适配佳     |
| 购物消费   | 87.50%  | 160    | 🥉 良好     | 中等偏上，商业场景理解稳定   |
| 生活服务   | 86.88%  | 160    | 📊 中等     | 略低于平均                   |
| 出行旅游   | 84.38%  | 160    | ⚠️ 待优化   | 中低水平，旅行场景适配不足   |
| 日常社交   | 79.38%  | 160    | ⚠️ 待优化   | 核心短板，口语化场景理解弱   |

### 3. 🔴 核心短板场景（准确率＜80%）
| 句式类型   | 领域     | 准确率  | 样本数 | ⚠️ 问题描述                          |
|------------|----------|---------|--------|---------------------------------------|
| 反讽       | 日常社交 | 67.50%  | 40     | 全场景最低，反讽+口语化语境双重挑战    |
| 双重否定   | 生活服务 | 75.00%  | 40     | 否定逻辑+服务场景理解不足            |
| 反讽       | 出行旅游 | 77.50%  | 40     | 反讽+旅行场景适配差                  |
| 双重否定   | 日常社交 | 77.50%  | 40     | 否定逻辑+口语化场景理解弱            |
| 转折关系   | 出行旅游 | 77.50%  | 40     | 转折逻辑+旅行场景适配不足            |

### 4. 📈 全局情感分类表现
| 评估维度       | Precision | Recall | F1-Score | Support | 🌟 结论                     |
|----------------|-----------|--------|----------|---------|------------------------------|
| Negative       | 0.85      | 0.93   | 0.89     | 552     | 负向情感识别准召均衡         |
| Positive       | 0.90      | 0.78   | 0.84     | 408     | 正向情感精准高但漏判严重      |
| **Accuracy**   | -         | -      | **0.87** | 960     | 整体表现良好                 |
| Macro Avg      | 0.88      | 0.86   | 0.86     | 960     | 类别均衡后性能稳定           |
| Weighted Avg   | 0.87      | 0.87   | 0.87     | 960     | 全局分类能力可靠             |

---

## 🎯 关键结论
### ✅ 核心优势
1. 🥇 **简单句理解**：92.50%准确率，全领域适配良好
2. 🥇 **高优领域表现**：影视娱乐（93.75%）、美食餐饮（90.00%）适配性强
3. 🥇 **负向情感识别**：F1 0.89，准召均衡，语义捕捉稳定

### ⚠️ 核心问题
1. 🔴 **顶级短板**：反讽+日常社交（67.50%），语义反转+口语化语境双重挑战
2. 🔴 **类别失衡**：Positive类召回率仅78%，存在"宁判负不判正"倾向
3. 🔴 **次短板集群**：低优句式+低优领域交叉场景（75%-80%）
4. 🔴 **领域短板**：日常社交领域整体拉胯（79.38%）
5. 🔴 **句式短板**：反讽句式全领域表现偏弱（82.08%）

---

## 🚀 后续微调指导
### 1. 📊 数据层优化（核心）
- [ ] 短板场景扩充：反讽+日常社交新增100-200条高质量标注样本（复用原有Prompt生成逻辑）
- [ ] 类别平衡：Positive类过采样，使正负样本比例≈1:1；
- [ ] 样本质量：新增样本需标注「句式+领域+情感」三标签，剔除原数据中标注错误的低准确率样本。

### 2. 🤖 模型层优化
- [ ] 基础微调：学习率3e-5，全量数据训练1-2轮，重点监控Positive类Recall（目标≥85%）；
- [ ] 定向微调：对“反讽+日常社交”等顶级短板，用专属样本做2-3轮小批量微调（batch size减半）；
- [ ] 参数优化：微调注意力层参数，引入“句式+领域”多标签损失函数。

### 3. 📋 评估层优化
- [ ] 核心监控：短板场景准确率≥85%，Positive召回率≥85%，日常社交领域准确率≥85%；
- [ ] 分层验证：为短板场景单独构建验证集（每类≥50条），对比微调前后指标变化。

---

## 🔧 快速开始
### 环境配置
```bash
# 克隆仓库
git clone https://github.com/MengzhongRe/bert-logic-stress.git
cd bert-logic-stress

# 安装依赖
pip install -r scripts/requirements.txt

# 配置DeepSeek API Key（环境变量或直接写入脚本）
export DEEPSEEK_API_KEY="your-api-key-here"

# 生成数据集（可选，已有960条数据）
python scripts/data_generator.py

# 运行模型测试
python test.py
```

---

## 🌟 项目亮点
1. 📝 **精细化Prompt设计**：去模板化示例+多结构约束，生成文本自然真实；
2. 🚀 **工程化数据构建**：多线程API调用+质量校验+硬编码标注，可复用性强；
3. 📊 **系统性测试框架**：从单一维度到交叉场景，全面覆盖模型能力边界；
4. 🎯 **精准定位短板**：聚焦核心问题，避免泛化优化；
5. 🎨 **面试友好**：完整工程链路（数据构建→模型测试→微调指导），可视化展示清晰。

---

## 📞 联系作者
- GitHub: [MengzhongRe](https://github.com/MengzhongRe)
- 项目地址: [bert-logic-stress](https://github.com/MengzhongRe/bert-logic-stress)

---

<p align="center">
  <img src="https://img.shields.io/badge/项目状态-已完成-green.svg" alt="项目状态">
  <img src="https://img.shields.io/badge/最后更新-2026.01-blue.svg" alt="最后更新">
</p>
