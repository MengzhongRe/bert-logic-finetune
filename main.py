# 项目根目录 main.py（唯一入口）
from scripts.model import load_model_and_tokenizer
from scripts.data_processor import build_datasets
from scripts.train import init_trainer, run_training
from scripts.utils import auto_detect_device

def main():
    """核心训练流程：设备检测→模型加载→数据构建→Trainer初始化→训练评估"""
    # 步骤1：自动检测训练设备（GPU/CPU）
    device = auto_detect_device()
    # 步骤2：加载预训练模型和分词器
    model, tokenizer = load_model_and_tokenizer(device)
    # 步骤3：构建训练/验证/测试数据集
    train_dataset, val_dataset, test_dataset = build_datasets(tokenizer, device)
    # 步骤4：初始化Hugging Face Trainer
    trainer = init_trainer(model, train_dataset, val_dataset, device)
    # 步骤5：执行训练+测试集评估+保存分词器
    run_training(trainer, test_dataset, tokenizer)

# 程序入口：仅直接运行时执行
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 训练执行出错：{str(e)}")
