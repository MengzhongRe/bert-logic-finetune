import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import warnings
# 新增：多线程相关库
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

warnings.filterwarnings('ignore')

# ================= 配置区（完全不变，保留你的所有设置）=================
# API配置
API_KEY = os.getenv('DEEPSEEK_API_KEY')
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-reasoner"

# ===== 核心配置：单句式单领域生成数量（建议5/10，保证配比精准）=====
COUNT_PER_TYPE_PER_DOMAIN = 40  # 单句式单领域生成N条 → 总条数=4句式×6领域×N条

# 情感/保存配置
SENTIMENT_LABELS = ["Positive", "Negative"]
SAVE_DIR = "data/big_data"
SAVE_FILENAME = os.path.join(SAVE_DIR, "deepseek_structured_data_cyclic_domain.csv")

# 反讽配比配置（单领域内按此配比）
IRONIC_POS_RATIO = 0.2
IRONIC_NEG_RATIO = 0.8
NORMAL_POS_RATIO = 0.5
NORMAL_NEG_RATIO = 0.5

# 6大核心领域（循环迭代使用）
DOMAINS = [
    "美食餐饮", "影视娱乐", "生活服务",
    "购物消费", "出行旅游", "日常社交"
]
# 逻辑句式
TARGET_TYPES = ["双重否定", "反讽", "转折关系", "简单句"]

# 初始化客户端（全局唯一，多线程共享）
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 系统提示词（完全不变）=================
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

# ================= Prompt 模版（完全不变）=================
def get_prompt(logic_type: str, domain: str, count: int) -> str:
    """
    生成指定「逻辑句式+领域」的用户提示词
    :param logic_type: 逻辑类型（反讽/简单句等）
    :param domain: 单一领域（美食餐饮/影视娱乐等）
    :param count: 单领域生成总条数
    :return: 格式化prompt
    """
    # 按句式计算单领域内的正负面数量（四舍五入，总数保证为count）
    if logic_type == "反讽":
        pos_count = round(count * IRONIC_POS_RATIO)
        neg_count = count - pos_count
        # 反讽的正面领域约束（和之前一致，避免AI生成不符合场景的正面反讽）
        pos_domain_tip = ""
        if pos_count > 0 and logic_type == "反讽" and domain not in ["美食餐饮", "购物消费", "日常社交"]:
            pos_domain_tip = "正面反讽需贴合熟人/亲友间的轻松亲昵场景，严格匹配该领域的日常表达；"
        prompt_base = f"""请生成{count}条【{logic_type}】逻辑句式、【{domain}】领域的中文日常口语化句子，其中{neg_count}条为Negative（负面），{pos_count}条为Positive（正面），**严格遵守数量配比，不可偏差、不可跨领域**。
【核心约束】
- 所有句子必须严格贴合{domain}领域，无跨领域内容，口语化、无AI生成感，长度8-30字；
- {pos_domain_tip}
- 反讽句式需严格遵循：负面字面褒扬实际贬斥，正面字面贬斥实际褒扬。
【示例参考（{logic_type}-{domain}）】"""
    else:
        pos_count = round(count * NORMAL_POS_RATIO)
        neg_count = count - pos_count
        prompt_base = f"""请生成{count}条【{logic_type}】逻辑句式、【{domain}】领域的中文日常口语化句子，其中{pos_count}条为Positive（正面），{neg_count}条为Negative（负面），**严格遵守数量配比，不可偏差、不可跨领域**。
【核心约束】
- 所有句子必须严格贴合{domain}领域，无跨领域内容，口语化、无AI生成感，长度8-30字；
【示例参考（{logic_type}-{domain}）】"""
    
    # 各类型-领域的示例（简洁版，聚焦领域特征，AI易模仿）
    if logic_type == '简单句':
        examples = {
            "美食餐饮": "- 周末和朋友去巷尾那家小馆，他们家的番茄牛腩炖得特别软烂，连汤汁都想拌米饭。(Positive)\n- 昨天点的酸菜鱼，鱼片吃着有股腥味，里面的配菜也不新鲜，没吃几口就扔了。(Negative)\n- 公司楼下的早餐摊，豆浆是现磨的，配着刚出锅的油条，早上吃着特别舒服。(Positive)",
            "影视娱乐": "- 昨晚在家看了部老电影，主角的演技特别自然，最后结局看得我眼眶都红了。(Positive)\n- 朋友推荐的那部网剧，剧情跳来跳去，演员台词也生硬，我勉强看了半集就关了。(Negative)\n- 新出的那个综艺，嘉宾们不刻意搞笑，就是日常聊天，反而觉得很真实。(Positive)",
            "生活服务": "- 上次寄快递，快递员看我搬不动箱子，主动帮我搬到车上，还提醒我记得保价。(Positive)\n- 上次约的家政阿姨，擦玻璃只擦了正面，窗台的灰都没清，说了她还不乐意。(Negative)\n- 小区门口的理发店，托尼老师会先问我的想法，剪完还教我怎么打理，很贴心。(Positive)",
            "购物消费": "- 网上买的那双帆布鞋，鞋底软软的，走了一天路也不磨脚，颜色和图片没差别。(Positive)\n- 之前买的那个保温杯，说是能保温12小时，结果下午水就凉了，根本达不到宣传的效果。(Negative)\n- 超市买的那个全麦面包，里面有很多坚果粒，吃着有嚼劲，饱腹感也强。(Positive)",
            "出行旅游": "- 上次去海边民宿，推开窗就能看到海，早上还能在阳台看日出，体验特别好。(Positive)\n- 跟团去的那个景区，导游一直在催着走，景点讲解也很敷衍，根本没好好玩。(Negative)\n- 上次自驾去山里，沿途的风景特别美，我们还在路边的农家乐吃了现摘的蔬菜。(Positive)",
            "日常社交": "- 我生日那天，闺蜜偷偷订了蛋糕，还叫了几个朋友来家里，真的特别惊喜。(Positive)\n- 之前约好和同事一起加班，结果他临时说有事要走，留下我一个人忙到半夜。(Negative)\n- 邻居阿姨知道我喜欢吃饺子，昨天包了一大盘送过来，还热乎着，特别暖心。(Positive)"
        }
    elif logic_type == '转折关系':
        examples = {
           "美食餐饮": "- 这家日料店位置有点偏，跟着导航找了好久，但刺身很新鲜，寿司的米也捏得刚好。(Positive)\n- 网上很火的那家网红奶茶，排队排了快半小时，喝着却很一般，茶底还有点涩。(Negative)\n- 虽然这家火锅锅底价格比别家贵，但分量足，涮肉的品质也高，算下来其实很值。(Positive)",
            "影视娱乐": "- 这部电影的开头有点慢，我差点快进，但越往后剧情越紧凑，反转也很意外。(Positive)\n- 演员阵容很强大，我本来很期待这部剧，可是剧情太老套，看了几集就没兴趣了。(Negative)\n- 虽然这个综艺没有流量明星，但嘉宾之间互动很自然，游戏环节也很有创意。(Positive)",
            "生活服务": "- 这家干洗店取衣服的时间比约定的晚了一天，但衣服洗得很干净，褶皱也都烫平了。(Positive)\n- 快递配送速度倒是很快，不过包裹外面破了个洞，里面的东西差点掉出来。(Negative)\n- 虽然这个健身房离我家有点远，但器材很全，教练也很专业，我还是愿意去。(Positive)",
            "购物消费": "- 这件外套价格不便宜，我犹豫了好久才买，但面料摸着很舒服，版型也很显身材。(Positive)\n- 这款耳机外观设计很好看，我很喜欢，可是续航时间很短，充一次电只能用3小时。(Negative)\n- 虽然这个包包容量不大，但设计很精致，搭配衣服很合适，我出门经常背。(Positive)",
            "出行旅游": "- 这个景区门票有点贵，但里面的景点很多，还有免费的讲解，玩下来觉得很充实。(Positive)\n- 酒店的装修很有特色，拍照很好看，可是房间隔音太差，晚上能听到隔壁的声音。(Negative)\n- 虽然这次旅行遇到了下雨，但我们在酒店一起打牌、聊天，反而觉得很开心。(Positive)",
            "日常社交": "- 我朋友平时话不多，看着有点高冷，但我遇到困难的时候，他总是第一个来帮我。(Positive)\n- 同事很热情，经常主动帮我带早餐，可是有时候会随便翻我的桌面，这点让我很不舒服。(Negative)\n- 虽然我和室友认识时间不长，但我们口味很合，经常一起做饭、看剧，相处得很愉快。(Positive)"
        }
    elif logic_type == '双重否定':
        examples = {
            "美食餐饮": "- 这家店的小笼包，咬开里面全是汤汁，不是没有一点不好吃，是真的没挑出毛病。(Positive)\n- 这家甜品店的蛋糕，甜得发腻，用料也不新鲜，没有一个人吃了不觉得失望的。(Negative)\n- 楼下那家面馆的牛肉面，肉给得足，汤也鲜，我吃了这么多次，从来没有觉得不好吃的时候。(Positive)",
            "影视娱乐": "- 这部纪录片拍得很真实，讲述的故事也很感人，看完没有不被打动的观众。(Positive)\n- 这部动画片的画风很粗糙，剧情也很无聊，小孩看了都没有不换台的。(Negative)\n- 那个老演员的演技，不管是哭戏还是笑戏，都特别有感染力，没有一场不让人佩服。(Positive)",
            "生活服务": "- 这个快递点的工作人员态度很好，包裹包装得也严实，我寄了好几次，没有一次不满意。(Positive)\n- 这家修鞋店的师傅手艺太差，鞋子越修越坏，顾客没有一个不抱怨的。(Negative)\n- 小区的保洁阿姨很负责，每天都把楼道打扫得干干净净，没有一处不整洁的地方。(Positive)",
            "购物消费": "- 这款洗发水洗完头发很顺滑，还带着淡淡的香味，用过的人没有不说好用的。(Positive)\n- 这家店的衣服质量太差，洗一次就变形，没有一件是能穿超过三次的。(Negative)\n- 这个牌子的薯片口感很脆，调味也刚刚好，我每次买，身边的人没有不抢着吃的。(Positive)",
            "出行旅游": "- 这家民宿的老板很热情，不仅给我们推荐景点，还帮我们订车票，没有一点不周到的地方。(Positive)\n- 这条旅游路线安排得太紧凑，每天都在赶路，游客没有一个不觉得累的。(Negative)\n- 这个古镇的风景很美，保留了很多老建筑，去过的人没有不称赞的。(Positive)",
            "日常社交": "- 我闺蜜很靠谱，答应我的事一定会做到，没有一次不兑现承诺的。(Positive)\n- 这个人说话不算数，经常放别人鸽子，跟他打交道的人没有不生气的。(Negative)\n- 我同事很热心，不管谁有困难他都愿意帮忙，公司里没有不夸他的。(Positive)"
        }
    elif logic_type == '反讽':
        examples = {
             "美食餐饮": "- 这家甜品店可真“过分”，芋泥冰给的料这么足，害得我吃完晚饭都吃不下了。(Positive)\n- 你可真会选地方，这家餐厅的菜又贵又难吃，我下次再也不想来了。(Negative)\n- 老板可真“小气”，送的小菜比我点的主菜还好吃，这不是让人多来几次嘛。(Positive)",
            "影视娱乐": "- 这部电影拍得可真“好”，我看了十分钟就开始打哈欠，差点在电影院睡着。(Negative)\n- 你推荐的这部剧可真“精彩”，剧情拖沓，演员演技尴尬，我都快进着看完的。(Negative)\n- 这个综艺可真“无聊”，嘉宾们聊的话题太有意思，我不知不觉就看了一整晚。(Positive)",
            "生活服务": "- 快递员可真“厉害”，把我的包裹放在小区门口，连个电话都不打，差点被别人拿错。(Negative)\n- 这家理发店的托尼老师可真“懂我”，我说剪短一点，他直接给我剪到了齐耳。(Negative)\n- 家政阿姨可真“偷懒”，把家里打扫得这么干净，我都找不到理由再请她来了。(Positive)",
            "购物消费": "- 你可真会“省钱”，买的这个包包质量这么好，背着这么好看，我都想跟你买同款了。(Positive)\n- 这款手机可真“耐用”，才用了半年，电池就不耐用了，充电还特别慢。(Negative)\n- 这家店可真“坑”，衣服价格这么便宜，质量还这么好，我一次买了三件。(Positive)",
            "出行旅游": "- 这家酒店可真“不错”，房间里有股霉味，窗户还打不开，住得我特别难受。(Negative)\n- 这个景区可真“值得”，门票贵，人又多，走了半天就看到几个普通的石头。(Negative)\n- 这次旅行可真“倒霉”，遇到的导游太负责，带我们玩了很多小众景点，还没花冤枉钱。(Positive)",
            "日常社交": "- 你可真“坏”，知道我喜欢吃草莓，特意买了一筐，还洗干净给我送来。(Positive)\n- 你可真“靠谱”，约好一起去看电影，我等了你半小时，你才说忘了。(Negative)\n- 我朋友可真“自私”，知道我加班，特意给我带了热乎的饭，还陪我一起加班。(Positive)"
        }
    # 获取对应领域的示例，无则用通用示例
    example_text = examples.get(domain, "- 正面句子：描述具体场景和真实的愉悦感受，不用固定结构。(Positive)\n- 负面句子：描述具体场景和真实的不满感受，不用固定结构。(Negative)\n- 中性句子：补充一个不同结构的示例，避免模板化。(Positive)")
    return prompt_base + example_text

# ================= 工具函数（完全不变）=================
def pre_check() -> bool:
    """程序运行前置校验"""
    if not API_KEY or API_KEY == "your_api_key_here":
        print("[ERROR] 请配置DEEPSEEK_API_KEY环境变量，或直接在配置区填入有效API_KEY")
        return False
    # 校验单领域数量：反讽需为5的倍数，保证8:2配比精准（5条=4负1正，10条=8负2正）
    if COUNT_PER_TYPE_PER_DOMAIN % 5 != 0 and IRONIC_POS_RATIO == 0.2:
        print(f"[WARNING] 反讽配比8:2，建议COUNT_PER_TYPE_PER_DOMAIN为5的倍数（5/10），保证单领域4负1正/8负2正")
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"[INFO] 已创建保存目录：{SAVE_DIR}")
    return True

def check_data_quality(items: list, logic_type: str, domain: str) -> list:
    """
    数据质量校验+硬编码标注type/domain/sub_type
    """
    valid_items = []
    for item in items:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        text = item.get("text").strip()
        if len(text) < 8 or len(text) > 50:
            continue
        label = item.get("label", "").strip()
        if label not in SENTIMENT_LABELS:
            continue
        # 硬编码赋值：type/domain 100%精准
        sub_type = "Normal"
        if logic_type == "反讽":
            sub_type = "Positive_Ironic" if label == "Positive" else "Negative_Ironic"
        # 直接组装所有字段，无需后续标注
        valid_items.append({
            "text": text,
            "label": label,
            "type": logic_type,
            "sub_type": sub_type,
            "domain": domain,
            "is_valid": True
        })
    print(f"[INFO] [{logic_type}-{domain}] 原始{len(items)}条，有效{len(valid_items)}条")
    return valid_items

# ================= 新增：多线程任务封装函数（核心）=================
def generate_single_task(logic_type: str, domain: str, count: int) -> list:
    """
    单任务封装：生成「单个句式+单个领域」的有效数据，供多线程调用
    :param logic_type: 逻辑句式
    :param domain: 领域
    :param count: 单领域生成数量
    :return: 该任务的有效数据列表，失败则返回空列表
    """
    try:
        # 调用API生成数据
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": get_prompt(logic_type, domain, count)}
            ],
            response_format={"type": "json_object"},
            temperature=1.3,
            max_tokens=8192,
            top_p=0.9
        )
        # 解析结果
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        data_dict = json.loads(content)
        raw_items = data_dict.get("items", [])
        # 质量校验+标注
        valid_items = check_data_quality(raw_items, logic_type, domain)
        return valid_items
    except json.JSONDecodeError as e:
        print(f"  [ERROR] [{logic_type}-{domain}] JSON解析失败：{str(e)[:80]}")
        return []
    except Exception as e:
        # 捕获所有异常，避免单任务失败影响线程池
        print(f"  [ERROR] [{logic_type}-{domain}] 生成失败：{str(e)[:80]}")
        return []

# ================= 主程序（多线程优化版，核心修改）=================
if __name__ == "__main__":
    if not pre_check():
        exit(1)
    
    # 计算总数，清晰展示计划
    TOTAL_DOMAIN = len(DOMAINS)
    TOTAL_TYPE = len(TARGET_TYPES)
    total_plan = TOTAL_TYPE * TOTAL_DOMAIN * COUNT_PER_TYPE_PER_DOMAIN
    all_data = []
    # 构建所有任务列表：(句式, 领域, 单领域生成数量)
    task_list = [
        (t_type, domain, COUNT_PER_TYPE_PER_DOMAIN)
        for t_type in TARGET_TYPES
        for domain in DOMAINS
    ]
    TOTAL_TASKS = len(task_list)
    print(f"\n[START] 多线程生成数据 | 模型: {MODEL_NAME} | 总任务数: {TOTAL_TASKS}")
    print(f"[CONFIG] 单句式单领域生成{COUNT_PER_TYPE_PER_DOMAIN}条 | 领域数{TOTAL_DOMAIN} | 句式数{TOTAL_TYPE}")
    print(f"[PLAN] 总计划生成：{TOTAL_TYPE}×{TOTAL_DOMAIN}×{COUNT_PER_TYPE_PER_DOMAIN} = {total_plan} 条")
    print(f"[RATIO] 配比规则：反讽单领域{IRONIC_NEG_RATIO*100:.0f}%负/{IRONIC_POS_RATIO*100:.0f}%正，其他50%正/50%负")
    print(f"[MULTI-THREAD] 启用线程池，并发数建议5-10（API限流友好）\n")

    # 多线程核心执行：创建线程池+提交任务+收集结果
    # 【关键】concurrency=8 可根据API限流调整，建议5-10，不要太大
    CONCURRENCY = 8
    start_time = time.time()  # 记录开始时间，统计耗时
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        # 提交所有任务，返回任务对象与参数的映射
        future_to_task = {
            executor.submit(generate_single_task, t, d, c): (t, d, c)
            for t, d, c in task_list
        }
        # 遍历完成的任务，收集结果（tqdm可视化进度）
        for future in tqdm(as_completed(future_to_task), total=TOTAL_TASKS, desc="多线程任务执行进度", ncols=80):
            task_result = future.result()  # 获取任务返回的有效数据
            if task_result:
                all_data.extend(task_result)

    # 计算耗时
    end_time = time.time()
    total_cost = round(end_time - start_time, 2)

    # 最终数据处理与保存（和原代码一致，无修改）
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.drop_duplicates(subset=["text"], keep="first")
        df = df.reset_index(drop=True)
        df = df[["text", "label", "type", "sub_type", "domain", "is_valid"]]
        df.to_csv(SAVE_FILENAME, index=False, encoding="utf-8-sig")

        # 打印详细统计
        total_actual = len(df)
        print(f"\n" + "="*80)
        print(f"[FINISH] 多线程生成完成！总耗时：{total_cost} 秒 | 平均单任务耗时：{round(total_cost/TOTAL_TASKS, 2)} 秒")
        print(f"[DATA] 计划{total_plan}条，实际保存{total_actual}条（有效率{total_actual/total_plan*100:.1f}%）")
        print(f"[SAVE] 保存至：{SAVE_FILENAME} | 字段：text/label/type/sub_type/domain/is_valid")
        print(f"[KEY] domain/type硬编码标注，准确率100% | 各领域数量绝对均等")
        print("="*80)

        # 核心统计：验证均等性
        print("\n【1. 各逻辑句式数据量统计（应均等）】")
        print(df["type"].value_counts().sort_index())
        print("\n【2. 6大领域数据量统计（应完全均等）】")
        domain_count = df["domain"].value_counts().sort_index()
        print(domain_count)
        print("\n【3. 句式×领域 交叉统计（验证每个组合数量均等）】")
        type_domain_cross = df.groupby(["type", "domain"]).size().unstack()
        print(type_domain_cross)
        # 反讽专项统计
        if "反讽" in df["type"].unique():
            ironic_df = df[df["type"] == "反讽"]
            print("\n【4. 反讽句式专项统计（8:2配比验证）】")
            print(f"反讽总条数：{len(ironic_df)}")
            print(ironic_df["label"].value_counts(normalize=True).map(lambda x: f"{x*100:.1f}%"))
            print("反讽各领域情感分布：")
            print(ironic_df.groupby("domain")["label"].value_counts())
    else:
        print(f"\n[ERROR] 未生成任何有效数据，耗时：{total_cost}秒，请检查API_KEY、网络或配置")