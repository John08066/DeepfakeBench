import os
import torch
from diffusers import AutoencoderKL
from PIL import Image
# --- ⬇⬇⬇ 核心修改点 ⬇⬇⬇ ---
# 我们不再导入 v2，而是导入旧的 v1 接口
import torchvision.transforms as transforms
# --- ⬆⬆⬆ 核心修改点 ⬆⬆⬆ ---
import argparse


def load_image(image_path, size=(256, 256)):
    """
    加载并预处理图像 (256x256)。
    归一化到 [-1, 1] 范围。
    """
    image = Image.open(image_path).convert("RGB")

    # 2. 定义预处理转换
    #    旧版 (v1) transforms 的写法
    preprocess = transforms.Compose(
        [
            # 旧版 Resize 写法，使用 PIL.Image.LANCZOS
            # (如果下面这行报错, 尝试 transforms.Resize(size), 去掉 interpolation)
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),  # 将 [0, 255] PIL 图像转为 [0.0, 1.0] 张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [0, 1] -> [-1, 1]
        ]
    )

    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor


def postprocess_image(image_tensor):
    """
    将 VAE 输出的张量转换回 PIL 图像。
    """
    image_tensor = image_tensor.squeeze(0)
    image_tensor = (image_tensor / 2) + 0.5
    image_tensor = image_tensor.clamp(0, 1)

    # 旧版 (v1) ToPILImage 的写法 (它是一个类)
    pil_image_converter = transforms.ToPILImage()
    image = pil_image_converter(image_tensor)

    return image


def main(args):
    """
    主函数：加载VAE，重建图像，并保存。
    """
    # 1. 设置设备
    if torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            print(f"错误: GPU ID {args.gpu_id} 超出了可用GPU数量 ({torch.cuda.device_count()})。")
            print(f"将使用默认的 GPU 0。")
            device = torch.device("cuda:0")
        else:
            device = torch.device(f"cuda:{args.gpu_id}")
    else:
        print("未检测到 CUDA，将使用 CPU。")
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # 2. 加载 VAE 模型
    vae_model_path = args.vae_path

    print(f"Loading VAE from: {vae_model_path} ...")
    try:
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(device)
    except Exception as e:
        print(f"错误：从 {vae_model_path} 加载 VAE 失败。")
        print(f"错误详情: {e}")
        return

    # 3. 设置为评估模式
    vae.eval()

    # 4. 加载和预处理图像
    print(f"Loading and preprocessing image: {args.input_path}")
    try:
        image_tensor = load_image(args.input_path).to(device)
    except FileNotFoundError:
        print(f"错误：找不到输入图片路径 {args.input_path}")
        return

    # 5. 通过 VAE 重建
    print("Reconstructing image through VAE...")
    with torch.no_grad():
        reconstructed_tensor = vae(image_tensor).sample

    # 6. 后处理图像
    print("Post-processing reconstructed image...")
    reconstructed_image = postprocess_image(reconstructed_tensor.cpu())

    # 7. 获取你指定的输出路径
    output_path = args.output_path

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 8. 保存重建的图像
    reconstructed_image.save(output_path)
    print(f"Reconstructed image saved to: {output_path}")


if __name__ == "__main__":
    # --- 设置命令行参数 ---
    parser = argparse.ArgumentParser(description="Reconstruct an image using a VAE.")

    # 输入图片路径 (必需)
    parser.add_argument(
        "-i", "--input_path",
        type=str,
        required=True,
        help="Path to the input image."
    )

    # VAE 模型路径 (可选, 有默认值)
    parser.add_argument(
        "-v", "--vae_path",
        type=str,
        default="/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/pretrained/vae",
        help="Path to the pretrained VAE directory."
    )

    parser.add_argument(
        "-o", "--output_path",
        type=str,
        required=True,  # <--- 改回 True
        help="Path to save the reconstructed image. (此为必需参数)"
    )

    # GPU ID (可选, 有默认值)
    parser.add_argument(
        "-g", "--gpu_id",
        type=int,
        default=0,
        help="ID of the GPU to use (e.g., 0, 1, 2...). 默认: 0"
    )

    args = parser.parse_args()
    main(args)