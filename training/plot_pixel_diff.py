import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import logging

# --- 导入你的自定义模块 ---
# 请确保你的 quant_pixel.py 放在 training 目录下
# 使得这些 import 路径能对应上
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
# 我们直接在这里重新定义一个精简版的 VAE 增强模块，避免复杂的 import
from diffusers import AutoencoderKL
import torch.nn as nn
import torch.nn.functional as F

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 1. 定义精简版 VAE 增强模块 (从你代码里复刻) ---
class VAEDataAugmentation_Simple(nn.Module):
    """
    精简版 VAE 重建模块，只负责输入->输出重建。
    """

    def __init__(self, vae_path):
        super(VAEDataAugmentation_Simple, self).__init__()
        self.vae_input_size = (256, 256)

        # 2. 加载 VAE
        logger.info(f"正在从本地加载 VAE: {vae_path}")
        try:
            # 推理建议使用 float16 节省显存
            self.vae = AutoencoderKL.from_pretrained(
                vae_path,
                torch_dtype=torch.float16
            )
        except Exception as e:
            logger.error(f"Failed to load VAE in float16: {e}. Trying float32.")
            self.vae = AutoencoderKL.from_pretrained(vae_path)

        # 3. 冻结 VAE
        self.vae.requires_grad_(False)
        self.vae.eval()

    @torch.no_grad()
    def forward(self, images):
        """
        输入: images [B, 3, H, W], float32 [0, 1] 范围
        输出: reconstructed [B, 3, H, W], float32 [0, 1] 范围
        """
        original_size = (images.shape[2], images.shape[3])
        vae_dtype = self.vae.dtype
        img_dtype = images.dtype

        # VAE 预处理: [0, 1] -> [-1, 1]
        x_vae_norm = (images - 0.5) / 0.5

        # Resize 到 (256, 256)
        x_vae_norm_resized = F.interpolate(
            x_vae_norm,
            size=self.vae_input_size,
            mode='bilinear',
            align_corners=False
        )

        # 3. VAE 重建
        vae_input = x_vae_norm_resized.to(vae_dtype)
        try:
            reconstructed = self.vae(vae_input).sample
        except RuntimeError:
            reconstructed = self.vae(vae_input).sample  # 偶尔的浮点数错误，重试

        # 转换精度: float16 -> float32
        reconstructed_f32 = reconstructed.to(img_dtype)

        # 4. 后处理：Resize 回原图 -> [-1, 1] -> [0, 1]
        reconstructed_resized = F.interpolate(
            reconstructed_f32,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )

        # 反归一化
        out_0_1 = (reconstructed_resized * 0.5) + 0.5
        # 防止数值截断错误，clamp 在 [0, 1] 之间
        out_0_1 = torch.clamp(out_0_1, 0, 1)

        return out_0_1


# --- 2. 核心计算与数据收集函数 ---
def collect_pixel_difference_metrics(config, vae_path, device):
    """
    运行数据加载器，通过 VAE 计算像素级差距，收集真假图片的 MSE 值。
    """

    # 初始化 VAE 模块
    vae_augmenter = VAEDataAugmentation_Simple(vae_path).to(device)

    # 准备数据加载器
    # 这里我们只取第一个测试数据集来做定量，避免太慢
    test_dataset_name = config['test_dataset'][0]
    config['test_dataset'] = test_dataset_name
    config['test_batchSize'] = 16  # VAE 显存占用高，batchSize 设小点

    logger.info(f"正在加载数据集: {test_dataset_name}, batchSize={config['test_batchSize']}")

    # 注意：确保这里调用的是 mode='test' 的数据加载，不包含额外的数据增强，
    # 并且 CLIP 的归一化参数要对上
    test_set = DeepfakeAbstractBaseDataset(
        config=config,
        mode='test',
    )
    test_data_loader = DataLoader(
        dataset=test_set,
        batch_size=config['test_batchSize'],
        shuffle=False,
        num_workers=4,  # 适当增加 worker，VAE 推理是瓶颈
        collate_fn=test_set.collate_fn,
        drop_last=False
    )

    # 用于 CLIP 的归一化/反归一化缓冲器 (必须使用 0-1 范围的 tensor)
    clip_mean = torch.tensor(config['mean']).view(1, -1, 1, 1).to(device)
    clip_std = torch.tensor(config['std']).view(1, -1, 1, 1).to(device)

    real_mses = []
    fake_mses = []

    total_batches = len(test_data_loader)
    logger.info("开始计算像素级 MSE...")

    for i, data_dict in tqdm(enumerate(test_data_loader), total=total_batches):
        images = data_dict['image'].to(device)  # [B, 3, 224, 224] (CLIP Normalization 分布)
        labels = torch.where(data_dict['label'] != 0, 1, 0)  # 0=Real, 1=Fake

        # 1. 像素级处理准备：将 CLIP Normalized 图片反归一化到 [0, 1] 范围
        images_0_1 = images * clip_std + clip_mean
        # 防止数值误差
        images_0_1 = torch.clamp(images_0_1, 0, 1)

        # 2. VAE 重建 (在 0-1 范围内进行)
        reconstructed_0_1 = vae_augmenter(images_0_1)

        # 3. 核心定量计算：图像层面均方误差 (MSE)
        # 计算每个样本的 MSE：(input - target)^2 的均值
        # 这里的 0-1 范围比 CLIP Normalized 范围在解释 MSE 时更直观
        mse_per_sample = torch.mean((images_0_1 - reconstructed_0_1) ** 2, dim=(1, 2, 3))  # 沿像素通道和宽高求均值

        # 4. 根据标签收集数据
        mse_np = mse_per_sample.cpu().numpy()
        real_mses.extend(mse_np[labels.cpu().numpy() == 0])
        fake_mses.extend(mse_np[labels.cpu().numpy() == 1])

        # 调试用：打印前几个样本看看数值量级
        # if i == 0:
        #     print(f"DEBUG: Batch MSE range: {mse_np.min():.6f} - {mse_np.max():.6f}")

    return np.array(real_mses), np.array(fake_mses)


# --- 3. 绘制密度分布图函数 (与特征差异图完全类似) ---
def plot_pixel_mse_distribution(real_mses, fake_mses, save_path='pixel_mse_distribution.png'):
    print("-" * 30)
    print(f"真实图像像素级 MSE: Count={len(real_mses)}, Mean={real_mses.mean():.6f}")
    print(f"伪造图像像素级 MSE: Count={len(fake_mses)}, Mean={fake_mses.mean():.6f}")
    print("-" * 30)

    # 美化设置
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    # 绘制 KDE 图
    # 注意：像素级 MSE 数值通常非常小（例如 1e-4），
    # 在 X 轴上，假图偏左（低误差，平滑），真图偏右（高误差，丢失细节）
    sns.kdeplot(fake_mses, color='#ff4d4f', fill=True, label='Fake (Generated)', alpha=0.5, linewidth=2.5)
    sns.kdeplot(real_mses, color='#1890ff', fill=True, label='Real (Natural)', alpha=0.5, linewidth=2.5)

    # 设置图表标题和标签
    plt.title('Pixel-level Reconstruction Error (MSE) Distribution (VAE)', fontsize=18)
    plt.xlabel('Pixel-wise Mean Squared Error (MSE)', fontsize=15)
    plt.ylabel('Density (Probability)', fontsize=15)

    # 添加图例和网格
    plt.legend(fontsize=15, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)

    # 限制 X 轴最小值为 0
    plt.xlim(left=0)

    # 对 X 轴使用科学计数法让界面不拥挤 (可选)
    # plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

    # 紧凑布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"图像层面 MSE 分布图已成功保存至: {save_path}")

    # plt.show() # 如果是本地运行


# --- 4. 主执行部分 ---
if __name__ == '__main__':
    # 路径设定 (请根据你服务器的实际路径修改)
    detector_config_path = '/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/config/detector/lora.yaml'
    test_config_path = '/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/config/test_config.yaml'

    # 这两个路径需要你在 lora.yaml 里找到并对应填在这里
    vae_path = '/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/pretrained/vae/'  # 指向包含 diffusion_pytorch_model.bin 的文件夹

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 整理 Config
    with open(detector_config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(test_config_path, 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)

    # 设定只跑一个数据集做演示，避免太久
    # 如果你要全跑，就改成 config['test_dataset'] 的全部内容，但时间会很长
    config['test_dataset'] = ["FaceForensics++"]  # 可以改成你觉得效果最明显的那个数据集

    logger.info(f"将在设备 {device} 上运行定量分析...")
    logger.info(f"目标数据集: {config['test_dataset'][0]}")

    # 2. 收集像素差异数据
    real_mses, fake_mses = collect_pixel_difference_metrics(config, vae_path, device)

    # 可选：如果你想保存这些 MSE 数据以便以后直接调画图，保存为另一个 PKL
    # pkl_save_path = '/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/pixel_mse.pkl'
    # with open(pkl_save_path, 'wb') as f:
    #     pickle.dump({'real_mse': real_mses, 'fake_mse': fake_mses}, f)
    # print(f"MSE数据已保存至 {pkl_save_path}")

    # 3. 绘图
    plot_pixel_mse_distribution(real_mses, fake_mses)