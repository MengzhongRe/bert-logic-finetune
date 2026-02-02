import pandas as pd
import os

# 原始数据集位路径
RAW_DATA1 = 'data/raw/train_data_v1.csv'
RAW_DATA2 = 'data/raw/train_data_v2.csv'
TEST_PATH = 'data/raw/test_golden.csv'
# 清洗后文件保存路径
RESULT_DIR = 'data/processed'
RESULT_NAME = 'base_data.csv'
# 读取两份文件为dataframe
df1 = pd.read_csv(RAW_DATA1,encoding='utf-8-sig')
df2 = pd.read_csv(RAW_DATA2,encoding='utf-8-sig')

print(f'第一分数据总数据量: {len(df1)}')
print(f'第二份数据总数据量: {len(df2)}')
# 合并两份数据集
combined_df = pd.concat([df1,df2],ignore_index=True)
print(f'合并后总数据量: {len(combined_df)}')
# 去重
combined_df = combined_df.drop_duplicates(subset=['text'],keep='first')
print(f'去重后，样本两位： {len(combined_df)}')
# 删除长度异常数值<=5或>35
combined_df = combined_df[
    (combined_df['text'].str.len() >= 6) &
    (combined_df['text'].str.len() <= 35)
]
print(f'删除异常长度后数据位: {len(combined_df)}')

# 提出训练集中已经在测试集中出现过的句子
if os.path.exists(TEST_PATH):
    test_data = pd.read_csv(TEST_PATH,encoding='utf-8-sig')
    test_sentences = set(test_data['text'].tolist())

    combined_df = combined_df[~combined_df['text'].isin(test_sentences)]
    print(f'剔除掉训练集在测试集中出现的数据之后总数据量为: {len(combined_df)}')

print(f'总数据句式分布为： {combined_df['type'].value_counts()}')
os.makedirs(RESULT_DIR,exist_ok=True)
combined_df.to_csv(os.path.join(RESULT_DIR,RESULT_NAME),index=False,encoding='utf-8-sig')