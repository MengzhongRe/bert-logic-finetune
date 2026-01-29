# scripts/manual_eval.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scripts.utils import auto_detect_device, cfg
# ✅ 核心：直接导入我们封装好的评估函数，复用代码
from scripts.evaluator import run_detailed_eval

# test_path = cfg['data']['manual_test_path']
# model_path = cfg["model"]["output_dir"]
model_path = 'uer/roberta-base-finetuned-dianping-chinese'
def main():
    print("🚀 启动独立评估模式 (无需重新训练)...")
    
    # 1. 准备环境
    device = auto_detect_device()

    
    print(f"📂 读取模型权重: {model_path}")
    
    try:
        # 2. 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path)
        model.to(device)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        print("请检查路径，或确认是否已运行 main.py 完成训练。")
        return

    # 3. 调用公共评估逻辑 (复用 scripts/evaluator.py)
    # 这样你只需要维护一份 BASELINE 配置
    run_detailed_eval(model, tokenizer, device)

if __name__ == "__main__":
    main()