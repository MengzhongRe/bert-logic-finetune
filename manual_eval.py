# scripts/manual_eval.py
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# 引入修改后的 evaluator
from scripts.evaluator import run_detailed_eval

# ================= 配置区 =================
# 指定 Hugging Face 模型 ID 或 本地路径
# MODEL_PATH = 'uer/roberta-base-finetuned-dianping-chinese'
MODEL_PATH = 'roberta_finetuned_model'
# 指定生成的测试数据路径 (请确保与生成文件路径一致)
TEST_DATA_PATH = 'data/train/test.csv'

# 指定结果输出目录
OUTPUT_DIR = 'eval_results_finetuned'
# ==========================================

def auto_detect_device():
    """简单的设备检测"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def main():
    print(f"🚀 启动评估脚本")
    print(f"🎯 目标模型: {MODEL_PATH}")
    print(f"📂 测试数据: {TEST_DATA_PATH}")
    
    # 1. 准备环境
    device = auto_detect_device()
    print(f"⚙️  运行设备: {device}")

    # 2. 检查数据文件
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ 找不到数据文件: {TEST_DATA_PATH}，请先运行数据生成脚本。")
        return

    try:
        # 3. 加载模型和分词器
        print("⏳ 正在加载模型权重 (首次运行可能需要下载)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 4. 调用评估逻辑
    run_detailed_eval(
        model=model, 
        tokenizer=tokenizer, 
        device=device, 
        test_path=TEST_DATA_PATH,
        output_dir=OUTPUT_DIR
    )

if __name__ == "__main__":
    main()