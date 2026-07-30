import os
import sys
import cv2
import torch
import numpy as np
import yaml
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from matplotlib import gridspec

# ======== 1. 路径设置 (请根据实际情况调整) ========
# 添加项目根目录到环境变量，确保能 import training.detectors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.detectors.lora_detector import LoraDetector    #clip+lora+vcae
from utils.grad_cam import GradCAM, ClassifierOutputTarget

from training.detectors.lora1_detector import Lora1Detector   # clip+lora

from training.detectors.clip_detector import CLIPDetector   # clip
#

# 权重路径
# /root/csy-7pw03c/disk/project/DeepfakeBench-main/training/logs/training/lora_2025-11-28-14-19-40/test/avg/ckpt_best.pth    # clip+lora
# /root/csy-7pw03c/disk/project/DeepfakeBench-main/training/logs/training/lora_2025-11-30-00-23-18/test/avg/ckpt_best.pth    # clip+lora+vae
# /root/csy-7pw03c/disk/project/DeepfakeBench-main/training/logs/training/csy/clip_2026-03-02-10-27-27/test/avg/ckpt_best.pth
weights_path = '/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/logs/training/csy/clip_2026-03-02-10-27-27/test/avg/ckpt_best.pth'
# 配置文件路径
config_path = "/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/config/detector/clip.yaml"

# /datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/000_003

# 图片列表
image_list = [
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/000_003/000.png",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/160_928/000.png",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/062_066/000.png",

    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/000_003/038.png",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/160_928/000.png",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/062_066/056.png",

    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/000_003/126.png",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/160_928/024.png",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/062_066/068.png",

    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/000_003/009.png",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/160_928/000.png",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/062_066/136.png",
]

# 对应的 Landmark 列表
landmark_list = [
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/landmarks/000_003/000.npy",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/landmarks/160_928/000.npy",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/landmarks/062_066/000.npy",

    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/landmarks/000_003/038.npy",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/landmarks/160_928/000.npy",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/landmarks/062_066/056.npy",

    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/landmarks/000_003/126.npy",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/landmarks/160_928/024.npy",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/landmarks/062_066/068.npy",

    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/landmarks/000_003/009.npy",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/landmarks/160_928/000.npy",
    "/datasets2/Deepfake/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/landmarks/062_066/136.npy",
]

# 标签列表 (假设全为伪造)
label_list = [1] * len(image_list)

# ======== 2. 设备与 CLIP 参数 ========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CLIP 标准均值和方差 (反归一化时必须用这个，否则颜色不对)
# CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
# CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CLIP_MEAN = [0.5, 0.5, 0.5]
CLIP_STD = [0.5, 0.5, 0.5]

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # CLIP 常用 224 或 336
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
])


# ======== 3. 核心工具函数 ========
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def load_image(image_path):
    # 使用 cv2 读取并转为 PIL，确保与 Transform 兼容
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    image_tensor = transform(pil).unsqueeze(0).to(device)
    return image_tensor


def load_landmark(landmark_path):
    if not os.path.exists(landmark_path):
        # 如果没有 landmark，生成一个全 0 的作为占位
        print(f"Warning: Landmark not found {landmark_path}, using zeros.")
        return torch.zeros((1, 81, 2)).to(device)
    lm = np.load(landmark_path).astype(np.float32)
    return torch.tensor(lm).unsqueeze(0).to(device)


def build_data_dict(image_list, landmark_list, label_list):
    imgs, lms = [], []
    for img_p, lm_p in zip(image_list, landmark_list):
        imgs.append(load_image(img_p))
        lms.append(load_landmark(lm_p))
    images = torch.cat(imgs, dim=0)
    landmarks = torch.cat(lms, dim=0)
    labels = torch.tensor(label_list, dtype=torch.long, device=device)
    return {'image': images, 'label': labels, 'landmark': landmarks, 'mask': None}


def load_model():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 实例化模型
    model = CLIPDetector(config)

    # 加载权重
    print(f"Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


# ======== 4. 关键：ViT 维度转换函数 ========
def clip_reshape_transform(tensor):
    """
    Grad-CAM 需要 4D tensor (B, C, H, W)，但 CLIP 输出的是 (B, Tokens, C)
    此函数负责将序列 reshape 回图像网格。
    """
    # 这里的 tensor 已经是 (Batch, Tokens, Channels)
    if tensor.ndim == 3:
        # 去掉 CLS token (通常是第0个)
        # CLIP L/14: 224x224 -> 256个patches + 1 CLS = 257 tokens
        cls_token = tensor[:, 0, :]
        image_tokens = tensor[:, 1:, :]

        num_tokens = image_tokens.shape[1]
        side = int(np.sqrt(num_tokens))

        # 简单检查
        if side * side != num_tokens:
            print(f"Warning: Token count {num_tokens} is not a perfect square.")

        # Reshape: (B, H*W, C) -> (B, H, W, C) -> (B, C, H, W)
        result = image_tokens.reshape(tensor.shape[0], side, side, tensor.shape[2])
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    return tensor


def _denorm_to_uint8(img_batch, mean, std):
    """
    反归一化，将 Tensor 转回可显示的 RGB numpy 数组
    """
    x = img_batch.detach().float().cpu().clone()

    # 适配维度进行广播
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)

    x = x * std + mean
    x = x.clamp(0, 1)
    # [B, 3, H, W] -> [B, H, W, 3]
    x = (x * 255.0).byte().permute(0, 2, 3, 1).numpy()
    return x


# ======== 5. 生成 Grad-CAM ========
def generate_gradcam(model, data_dict):
    model.eval()
    data_dict['image'].requires_grad_(True)

    # 路径：LoraDetector -> backbone(PeftModel) -> base_model(LoraModel) -> model(CLIP) -> encoder
    # 选取最后一层 Encoder Block 的 LayerNorm1 (Attention之前)
    target_layers = [
        # model.backbone.base_model.model.encoder.layers[21].layer_norm1,
        # model.backbone.base_model.model.encoder.layers[22].layer_norm1,
        # model.backbone.base_model.model.encoder.layers[23].layer_norm1,
        model.backbone.encoder.layers[23].layer_norm1,
    ]

    # 初始化 GradCAM，必须传入 reshape_transform
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=clip_reshape_transform  # <--- 必须加这个
    )

    labels_cpu = data_dict['label'].detach().cpu().tolist()
    targets = [ClassifierOutputTarget(int(y)) for y in labels_cpu]

    # 生成热力图
    grayscale_cam = cam(data_dict=data_dict, targets=targets)

    # 格式统一处理
    if isinstance(grayscale_cam, torch.Tensor):
        grayscale_cam = grayscale_cam.detach().cpu().numpy()
    elif isinstance(grayscale_cam, list):
        grayscale_cam = np.stack(grayscale_cam, axis=0)
    if grayscale_cam.ndim == 2:
        grayscale_cam = grayscale_cam[None, ...]

    return grayscale_cam


# ======== 6. 可视化展示 ========
def show_cam_grid(original_rgb_uint8, cams, image_weight=0.6, mode='overlay'):
    B, H, W, _ = original_rgb_uint8.shape
    rows = 1

    # 每组3张
    images_per_group = 3
    num_groups = (B + images_per_group - 1) // images_per_group

    total_cols = num_groups * images_per_group + max(0, num_groups - 1)

    # 动态计算图表大小
    fig = plt.figure(figsize=(3.5 * images_per_group * num_groups, 3.5))

    width_ratios = []
    for g in range(num_groups):
        width_ratios.extend([1] * images_per_group)
        if g < num_groups - 1:
            width_ratios.append(0.2)  # 组间空隙

    gs = gridspec.GridSpec(rows, total_cols, width_ratios=width_ratios, wspace=0.0, hspace=0.0)

    for i in range(B):
        img_rgb = original_rgb_uint8[i]
        cam_i = cams[i]

        # 缩放热力图到原图尺寸
        if cam_i.shape[:2] != (H, W):
            cam_i = cv2.resize(cam_i, (W, H), interpolation=cv2.INTER_LINEAR)

        # 制作叠加图
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_i), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        overlay = image_weight * img_rgb.astype(np.float32) + (1.0 - image_weight) * heatmap_rgb.astype(np.float32)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # 计算子图位置
        group_idx = i // images_per_group
        within_group_idx = i % images_per_group
        col_offset = group_idx * (images_per_group + 1)
        col_idx = col_offset + within_group_idx

        ax = fig.add_subplot(gs[0, col_idx])

        if mode == 'orig':
            ax.imshow(img_rgb)
        elif mode == 'overlay':
            ax.imshow(overlay)
        ax.axis('off')

    plt.tight_layout(pad=0.0)
    plt.show()


# ======== 7. 主函数 ========
def main():
    # 1. 加载模型
    model = load_model()

    # 2. 构建数据
    if not image_list:
        print("错误：图片列表为空，请检查路径。")
        return

    data_dict = build_data_dict(image_list, landmark_list, label_list)
    set_seed(1024)

    # 3. 生成 Grad-CAM
    print("Generating Grad-CAM...")
    cams = generate_gradcam(model, data_dict)

    # 4. 反归一化原图 (使用 CLIP 的 mean/std)
    imgs_rgb = _denorm_to_uint8(data_dict['image'], mean=CLIP_MEAN, std=CLIP_STD)

    # 5. 展示
    print("Displaying results...")
    show_cam_grid(imgs_rgb, cams, image_weight=0.6, mode='overlay')


if __name__ == "__main__":
    main()