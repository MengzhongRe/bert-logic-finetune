import pandas as pd

# DATA_DIR = 'data/raw/deepseek_structured_data_cyclic_domain_960_v'
SAVE_DIR = 'data/processed'

# df1 = pd.read_csv(f'{DATA_DIR}1.csv',encoding='utf-8-sig')
# df2 = pd.read_csv(f'{DATA_DIR}2.csv',encoding='utf-8-sig')
# df3 = pd.read_csv(f'{DATA_DIR}3.csv',encoding='utf-8-sig')

# combined_raw = pd.concat([df1,df2,df3],ignore_index=True)
# print(f'合并后样本总数量: {len(combined_raw)}')

# combined_raw = combined_raw.drop_duplicates(subset=['text'],keep='first')
# print(f'去重后的样本数: {len(combined_raw)}')

RESULT_DIR = 'data/test_result'
RESULT_NAME = 'logic_test_result.csv'

pred_df1 = pd.read_csv(f'{RESULT_DIR}/v1/{RESULT_NAME}',encoding='utf-8-sig')
pred_df2 = pd.read_csv(f'{RESULT_DIR}/v2/{RESULT_NAME}',encoding='utf-8-sig')
pred_df3 = pd.read_csv(f'{RESULT_DIR}/v3/{RESULT_NAME}',encoding='utf-8-sig')

all_pred = pd.concat([pred_df1,pred_df2,pred_df3],ignore_index=True)

correct_pred = all_pred[all_pred['label'] == all_pred['Pred_Label']]
print(f'三次测试中模型预测正确的样本数: {len(correct_pred)}')

# 进一步筛选“高准确率场景”的样本（句式+领域组合准确率≥85%）
# 先计算每个句式+领域的准确率
scene_acc = correct_pred.groupby(['type','domain']).size() / all_pred.groupby(['type','domain']).size()
high_acc_scenes = scene_acc[scene_acc >= 0.87].index.tolist()

# 筛选高准确率场景的样本
high_acc_samples = correct_pred[
    correct_pred.apply(lambda x: (x["type"], x["domain"]) in high_acc_scenes, axis=1)
]
print("高准确率样本数（基础训练数据）： ",len(high_acc_samples))

# ===================== 步骤5：过滤文本长度异常（6-30字）=====================
high_acc_samples = high_acc_samples[
    (high_acc_samples['text'].str.len() >= 6) &
    (high_acc_samples['text'].str.len() <= 35)
]
print('过滤长度异常后有效样本数： ',len(high_acc_samples))
# ===================== 步骤6：处理sub_type（核心：保留所有合法取值，无过滤）=====================
print('\nsub_type字段分布：')
print(high_acc_samples['sub_type'].value_counts())

# ===================== 步骤7：统一情感标签格式（0=Negative，1=Positive）=====================
high_acc_samples['label'] = high_acc_samples['label'].map({
    'Negative': 0,
    'Positive': 1,
})
print('\n转换后标签情感分布：')
print(high_acc_samples['label'].value_counts())

# ===================== 步骤8：精简核心字段，删除冗余字段，标记有效样本 =====================
core_fileds = ['text','label','type','sub_type','domain']
high_acc_samples = high_acc_samples[core_fileds]
# 标记为有效样本，便于后续和新增短板样本合并
high_acc_samples['is_valid'] = True

# ===================== 最终结果输出 =====================
print('\n最终筛选处的高优数据，一共',len(high_acc_samples))
print('\n最终数据前五行预览')
print(high_acc_samples.head())

# ===================== 保存最终基础数据（用于后续合并新增短板样本）=====================
high_acc_samples.to_csv(f'{SAVE_DIR}/high_acc_base_samples.csv',index=False,encoding='utf-8-sig')
print(f'高优基础样本已保存至: {SAVE_DIR}/high_acc_base_samples.csv')

#======================与新增的短板数据合并============================
# 加载基础样本和新增短板样本
base_df = pd.read_csv("data/processed/high_acc_base_samples.csv", encoding="utf-8-sig")
new_shortage_df = pd.read_csv("data/raw/new_shortage_samples.csv", encoding="utf-8-sig")
# 合并
combined_df = pd.concat([base_df, new_shortage_df], ignore_index=True)
# 去重（保险起见）
combined_df = combined_df.drop_duplicates(subset=["text"], keep="first")
print(f"基础样本+新增短板样本合并后总数量：{len(combined_df)}")
# 保存合并后数据，用于后续类别平衡和分割
combined_df.to_csv("data/processed/combined_all_samples.csv", index=False, encoding="utf-8-sig")