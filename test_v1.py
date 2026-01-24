from transformers import pipeline
import torch
import pandas as pd

checkpoint = 'uer/roberta-base-finetuned-dianping-chinese'
print(f'正在加载模型: {checkpoint}...')
classifier = pipeline('sentiment-analysis',model=checkpoint)

test_data = [
    # --- A. 双重否定 (Double Negation) ---
    {"text": "我不能不同意你的观点。", "label": "Positive", "type": "双重否定"},
    {"text": "这部电影不是不精彩。", "label": "Positive", "type": "双重否定"},
    {"text": "我不讨厌这个设计。", "label": "Positive", "type": "双重否定"},
    
    # --- B. 转折/让步 (Concessive) ---
    {"text": "虽然服务员很热情，但是菜太难吃了。", "label": "Negative", "type": "转折关系"},
    {"text": "虽然价格很贵，但是物超所值。", "label": "Positive", "type": "转折关系"},
    
    # --- C. 反讽 (Irony/Sarcasm) - BERT 的死穴 ---
    {"text": "你可真聪明，把我的系统都搞崩了。", "label": "Negative", "type": "反讽"},
    {"text": "感谢你在百忙之中敷衍我。", "label": "Negative", "type": "反讽"},
    {"text": "这排队体验简直太‘棒’了，等了两个小时。", "label": "Negative", "type": "反讽"},
    
    # --- D. 简单句 (对照组) ---
    {"text": "味道很好。", "label": "Positive", "type": "简单句"},
    {"text": "体验很差。", "label": "Negative", "type": "简单句"}
]

#3.运行测试
print('开始行为测试 (Behavior testing)...')
results = []
for item in test_data:
    #预测
    pred = classifier(item['text'])[0]

    #映射标签
    model_label = pred['label']
    score = pred['score']

    #简单的后处理
    is_positive = 'positive' in model_label
    pred_sentiment = 'Positive' if is_positive else 'Negative'

    results.append({
        '句子':item['text'],
        '类型':item['type'],
        '真实标签':item['label'],
        '预测标签':pred_sentiment,
        '置信度':round(score,4),
        '是否正确':'✅' if item['label'] == pred_sentiment else '❌'
    })

df = pd.DataFrame(results)

#打印表格
print('\n' + '='*30 + 'BERT 逻辑评估能力结果' + '='*30)
print(df.to_markdown(index=False))

#统计各类别的准确率
print('\n=====错误率分析=====')
analysis = df.groupby('类型')['是否正确'].apply(lambda x: (x == '✅').mean()).reset_index()
analysis.columns = ['类型','准确率']
print(analysis.to_markdown(index=False))



