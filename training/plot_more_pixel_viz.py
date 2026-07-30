import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # 画分布图需要用到
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging

# --- 导入模块 ---
try:
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    from diffusers import AutoencoderKL
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    print(f"导入失败: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 1. 完全复刻的 VAE 模块 (无任何修改) ---
class VAEDataAugmentation_Exact(nn.Module):
    def __init__(self, vae_path, clip_mean, clip_std):
        super(VAEDataAugmentation_Exact, self).__init__()
        self.vae_input_size = (256, 256)

        self.register_buffer('clip_mean', clip_mean.view(1, -1, 1, 1))
        self.register_buffer('clip_std', clip_std.view(1, -1, 1, 1))

        vae_mean = torch.tensor([0.5, 0.5, 0.5])
        vae_std = torch.tensor([0.5, 0.5, 0.5])
        self.register_buffer('vae_mean', vae_mean.view(1, -1, 1, 1))
        self.register_buffer('vae_std', vae_std.view(1, -1, 1, 1))

        logger.info(f"Loading VAE: {vae_path}")
        try:
            self.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
        except:
            self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.vae.requires_grad_(False)
        self.vae.eval()

    def clip_denormalize(self, x):
        return x * self.clip_std + self.clip_mean

    def vae_normalize(self, x):
        return (x - self.vae_mean) / self.vae_std

    def vae_denormalize(self, x):
        return x * self.vae_std + self.vae_mean

    @torch.no_grad()
    def forward(self, images):
        original_size = (images.shape[2], images.shape[3])
        x = self.clip_denormalize(images)
        x_vae_norm = self.vae_normalize(x)
        x_vae_norm_resized = F.interpolate(x_vae_norm, size=self.vae_input_size, mode='bilinear', align_corners=False)

        vae_input = x_vae_norm_resized.to(self.vae.dtype)
        try:
            reconstructed = self.vae(vae_input).sample
        except RuntimeError:
            return images

        reconstructed_f32 = reconstructed.to(images.dtype)
        reconstructed_resized = F.interpolate(reconstructed_f32, size=original_size, mode='bilinear',
                                              align_corners=False)
        out_0_1 = self.vae_denormalize(reconstructed_resized)
        return torch.clamp(out_0_1, 0, 1)


# --- 2. 基于 Landmark 的人脸区域差异分布计算 ---
@torch.no_grad()
def run_face_region_distribution(config, vae_path, device):
    # 为了加快统计速度，我们可以把 batchSize 调大一点 (比如 8 或 16，取决于你的显存)
    config['test_batchSize'] = 16

    target_dataset = config['test_dataset'][0] if isinstance(config['test_dataset'], list) else config['test_dataset']
    config['test_dataset'] = target_dataset

    clip_mean_tensor = torch.tensor(config['mean']).to(device)
    clip_std_tensor = torch.tensor(config['std']).to(device)

    vae_augmenter = VAEDataAugmentation_Exact(vae_path, clip_mean_tensor, clip_std_tensor).to(device)

    test_set = DeepfakeAbstractBaseDataset(config=config, mode='test')
    test_data_loader = DataLoader(dataset=test_set, batch_size=config['test_batchSize'], shuffle=False,
                                  collate_fn=test_set.collate_fn)

    real_mses = []
    fake_mses = []

    logger.info("开始遍历数据，计算人脸区域的重构误差 (MSE)...")

    for data_dict in tqdm(test_data_loader):
        img_tensor = data_dict['image'].to(device)
        labels = data_dict['label']

        # 确保 dataloader 成功读取了 landmark
        if 'landmark' not in data_dict or data_dict['landmark'] is None:
            logger.error("未找到 landmark 数据！请检查 config 中的 with_landmark 是否生效。")
            break

        landmarks = data_dict['landmark']  # 形状通常为 [B, num_points, 2]

        # 1. 获取原图和重构图 [0, 1]
        orig_0_1 = torch.clamp(img_tensor * clip_std_tensor.view(1, -1, 1, 1) + clip_mean_tensor.view(1, -1, 1, 1), 0,
                               1)
        reco_0_1 = vae_augmenter(img_tensor)

        # 2. 遍历 Batch 中的每一张图，利用 Landmark 裁剪出人脸 Bounding Box
        B, C, H, W = orig_0_1.shape
        for b in range(B):
            label = labels[b].item()
            lm = landmarks[b].cpu().numpy()  # [num_points, 2]

            # 提取 landmark 的最小包围盒 (Bounding Box)
            # lm[:, 0] 是 x 坐标 (宽)，lm[:, 1] 是 y 坐标 (高)
            x_min, x_max = int(np.min(lm[:, 0])), int(np.max(lm[:, 0]))
            y_min, y_max = int(np.min(lm[:, 1])), int(np.max(lm[:, 1]))

            # 防止坐标越界
            x_min, x_max = max(0, x_min), min(W, x_max)
            y_min, y_max = max(0, y_min), min(H, y_max)

            # 如果 landmark 异常导致面积为 0，跳过该样本
            if x_max <= x_min or y_max <= y_min:
                continue

            # 3. 【核心】只截取人脸区域计算 MSE
            face_orig = orig_0_1[b, :, y_min:y_max, x_min:x_max]
            face_reco = reco_0_1[b, :, y_min:y_max, x_min:x_max]

            # 计算该人脸区域的均方误差
            mse = torch.mean((face_orig - face_reco) ** 2).item()

            if label == 0:
                real_mses.append(mse)
            else:
                fake_mses.append(mse)

    # --- 3. 绘制分布图 (KDE) ---
    real_mses = np.array(real_mses)
    fake_mses = np.array(fake_mses)

    logger.info(f"统计完毕！Real 样本数: {len(real_mses)}, 平均人脸区域 MSE: {real_mses.mean():.6f}")
    logger.info(f"统计完毕！Fake 样本数: {len(fake_mses)}, 平均人脸区域 MSE: {fake_mses.mean():.6f}")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # 绘制平滑的核密度估计图
    sns.kdeplot(fake_mses, color='#ff4d4f', fill=True, label='Fake (Face Region)', alpha=0.5, linewidth=2)
    sns.kdeplot(real_mses, color='#1890ff', fill=True, label='Real (Face Region)', alpha=0.5, linewidth=2)

    plt.title('Distribution of VAE Reconstruction MSE (Face Region Only)', fontsize=16)
    plt.xlabel('Pixel-wise Mean Squared Error within Landmark Bounding Box', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=0)  # 误差不可能小于0

    # --- 保存图片 ---
    save_dir = '/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/pic'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'face_region_mse_distribution.png')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    logger.info(f"人脸区域差异分布图保存完毕: {save_path}")


if __name__ == '__main__':
    detector_config_path = '/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/config/detector/lora.yaml'
    test_config_path = '/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/config/test_config.yaml'
    vae_path = '/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/pretrained/vae/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(detector_config_path, 'r') as f: config = yaml.safe_load(f)
    with open(test_config_path, 'r') as f: config.update(yaml.safe_load(f))

    run_face_region_distribution(config, vae_path, device)