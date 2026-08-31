import torch
import torch.nn as nn
from detectors.lora_detector import LoraDetector  # 请根据你的项目路径调整 import


def count_parameters(model):
    # 总参数 (包含冻结的 VAE 和 CLIP 骨干)
    total_params = sum(p.numel() for p in model.parameters())

    # 真正参与训练的参数 (LoRA 层 + 分类头 Head)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 专门看分类头 Head 的参数
    head_params = sum(p.numel() for p in model.head.parameters())

    # 计算 LoRA 部分的参数量 (从 backbone 中提取)
    # 注意：backbone 已经被 get_peft_model 包装过
    lora_params = trainable_params - head_params

    return total_params, trainable_params, lora_params, head_params


def main():
    # 1. 模拟你的 config (根据你的代码需求补全必要字段)
    config = {
        'mean': [0.48145466, 0.4578275, 0.40821073],  # CLIP 标准均值
        'std': [0.26862954, 0.26130258, 0.27577711],  # CLIP 标准方差
        'vae_path': "/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/pretrained/vae",  # 你的 VAE 路径
    }

    # 2. 实例化模型
    print("正在初始化模型并注入 LoRA...")
    model = LoraDetector(config=config)

    # 3. 加载权重 (可选，如果不加载，初始化的参数量也是一样的)
    ckpt_path = "/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/logs/training/lora_2025-11-30-00-23-18/test/avg/ckpt_best.pth"
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # 如果保存的是整个 model state_dict
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"成功加载权重: {ckpt_path}")
    except Exception as e:
        print(f"未加载权重 (仅计算结构参数): {e}")

    # 4. 执行统计
    total, trainable, lora, head = count_parameters(model)

    print("\n" + "=" * 40)
    print(f"模型参数统计结果:")
    print("-" * 40)
    print(f"总参数量 (Total):      {total:,}")
    print(f"可训练参数 (Trainable): {trainable:,}")
    print("-" * 40)
    print(f"其中 LoRA 引入参数:     {lora:,}")
    print(f"其中分类头 (Head):      {head:,}")
    print(f"训练参数占比:           {(trainable / total) * 100:.4f}%")
    print("=" * 40)

    # 5. 使用 peft 内置方法双重验证 (仅针对 backbone)
    print("\nBackbone (PEFT) 详细统计:")
    model.backbone.print_trainable_parameters()


if __name__ == "__main__":
    main()