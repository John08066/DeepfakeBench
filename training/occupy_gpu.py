import torch
import sys
import time

def occupy_gpu_memory(device_id=0, size_in_gb=10, hold_time=None):
    """
    占用指定GPU显存
    :param device_id: GPU编号
    :param size_in_gb: 要占用的显存大小（GB）
    :param hold_time: 持续时间（秒），None表示手动释放
    """
    device = torch.device(f"cuda:{device_id}")
    print(f"🚀 尝试在 {device} 上占用 {size_in_gb} GB 显存...")

    num_elements = int(size_in_gb * (1024 ** 3) / 4)  # float32=4字节
    try:
        tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
        print(f"✅ 成功占用约 {tensor.numel() * 4 / 1024 ** 3:.2f} GB 显存。")

        if hold_time is not None:
            print(f"⏳ 将保持 {hold_time} 秒后自动释放...")
            time.sleep(hold_time)
            del tensor
            torch.cuda.empty_cache()
            print("💨 已自动释放显存。")
        else:
            input("按 Enter 释放显存并退出...")
    except RuntimeError as e:
        print("❌ 分配失败：显存可能不足。错误信息：", e)

if __name__ == "__main__":
    # 从命令行参数读取显存大小和GPU编号
    # 用法: python occupy_gpu.py <size_in_gb> [device_id] [hold_time]
    size_in_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    hold_time = float(sys.argv[3]) if len(sys.argv) > 3 else None

    occupy_gpu_memory(device_id=device_id, size_in_gb=size_in_gb, hold_time=hold_time)
