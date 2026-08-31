import os
import cv2
import torch
import numpy as np
import yaml
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from utils.grad_cam import GradCAM, ClassifierOutputTarget

# ======== 设备 ========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== 加载你项目内的模型类 ========
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from training.detectors.aour_detector import AourDetector  # 模型类路径
# from training.detectors.xception_detector import XceptionDetector
from training.detectors.lora_detector import LoraDetector

# ======== 路径与数据 ========
# weights_path = "/home/csy/disk1/project_1/deepfakeBench/DeepfakeBench-main/training/logs/training/aour1_2025-06-19-18-28-42/test/avg/ckpt_best.pth"
# weights_path = "/home/csy/disk1/project_1/deepfakeBench/DeepfakeBench-main/training/logs/training/xception_2024-11-16-10-33-42/test/avg/ckpt_best.pth"
# weights_path = "/home/csy/disk1/project_1/deepfakeBench/DeepfakeBench-main/training/weights/xception_best.pth"
weights_path = '/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/logs/training/lora_2025-11-30-00-23-18/test/avg/ckpt_best.pth'


image_list = [
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/000_003/000.png",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/160_928/000.png",
    # "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/276_185/234.png",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/062_066/000.png",

    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/000_003/038.png",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/160_928/000.png",
    # "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/276_185/056.png",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/062_066/056.png",

    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/000_003/126.png",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/160_928/024.png",
    # "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/276_185/000.png",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/062_066/068.png",

    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/000_003/009.png",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/160_928/012.png",
    # "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/276_185/021.png",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/062_066/136.png",

]
landmark_list = [
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/landmarks/000_003/000.npy",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/landmarks/160_928/000.npy",
    # "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/landmarks/276_185/234.npy",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/landmarks/062_066/000.npy",

    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/landmarks/000_003/038.npy",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/landmarks/160_928/000.npy",
    # "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/landmarks/276_185/056.npy",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/landmarks/062_066/056.npy",

    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/landmarks/000_003/126.npy",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/landmarks/160_928/024.npy",
    # "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/landmarks/276_185/000.npy",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/landmarks/062_066/068.npy",

    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/landmarks/000_003/009.npy",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/landmarks/160_928/012.npy",
    # "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/landmarks/276_185/021.npy",
    "/home/csy/disk1/dataset/DeepfakeBench/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/landmarks/062_066/136.npy",

]


label_list = [1,1,1,1,1,1,1,1,1,1,1,1]

# label_list = [1]

# ======== 图像预处理（注意：Resize 需要 PIL 或 Tensor）=======
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                          std=[0.5, 0.5, 0.5])
# ])

# clip
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])


def set_seed(seed, use_cuda=True):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    # torch.use_deterministic_algorithms(True)

def load_image(image_path):
    """
    读取单张图像并预处理 -> [1,3,256,256] (GPU)
    关键修复：把 numpy 转成 PIL 再传给 transform，避免 Resize 报错。
    """
    img = cv2.imread(image_path)               # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB numpy(H,W,3)
    pil = Image.fromarray(img)                 # 转 PIL
    image_tensor = transform(pil).unsqueeze(0).to(device)
    return image_tensor

def load_landmark(landmark_path):
    lm = np.load(landmark_path).astype(np.float32)
    return torch.tensor(lm).unsqueeze(0).to(device)

def build_data_dict(image_list, landmark_list, label_list):
    """
    打包 batch 字典：
      - image:   FloatTensor [B,3,256,256] (GPU)
      - label:   LongTensor  [B]          (GPU)
      - landmark:FloatTensor [B,...]      (GPU)
      - mask:    None（占位，和参考实现一致）
    """
    assert len(image_list) == len(landmark_list) == len(label_list), "列表长度不一致"
    assert len(image_list) > 0, "列表不能为空"

    imgs, lms = [], []
    for img_p, lm_p in zip(image_list, landmark_list):
        imgs.append(load_image(img_p))
        lms.append(load_landmark(lm_p))
    images = torch.cat(imgs, dim=0)
    landmarks = torch.cat(lms, dim=0)
    labels = torch.tensor(label_list, dtype=torch.long, device=device)

    return {'image': images, 'label': labels, 'landmark': landmarks, 'mask': None}

def load_model():
    """加载模型到 GPU 并 eval。"""
    # config_path = "/home/csy/disk1/project_1/deepfakeBench/DeepfakeBench-main/training/config/detector/aour.yaml"   # xception.yaml
    config_path = "/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/config/detector/lora.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # model = XceptionDetector(config)
    # model = AourDetector(config)
    model = LoraDetector(config)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    # 确认 crossfusion 存在且可用于 Grad-CAM
    # assert hasattr(model, 'crossfusion'), "model.crossfusion 不存在，请确认模型实现。"
    # assert isinstance(model.crossfusion, torch.nn.Module), "model.crossfusion 必须是 nn.Module。"
    return model

def _denorm_to_uint8(img_batch, mean=0.5, std=0.5):
    """
    将 (x-mean)/std 的 [B,3,H,W] 反归一化 -> [B,H,W,3] uint8 (RGB)
    """
    x = img_batch.detach().float().cpu().clone()
    x = x * std + mean
    x = x.clamp(0, 1)
    x = (x * 255.0).byte().permute(0, 2, 3, 1).numpy()
    return x

def generate_gradcam(model, data_dict):
    """
    严格复用参考实现的 Grad-CAM 调用方式：
      - target_layers = [model.crossfusion]
      - targets = [ClassifierOutputTarget(label_i), ...]
      - cam(data_dict=data_dict, targets=targets)
    返回：numpy ndarray，形状 [B,H,W]，每张一个灰度热力图。
    """
    model.eval()
    # 与参考实现一致：允许输入图像参与梯度回传
    data_dict['image'].requires_grad_(True)

    target_layers = [

        # aour1

        # stream1
        # model.backbone_stream1.block1,
        # model.backbone_stream1.block2,
        # model.backbone_stream1.block3,
        # model.backbone_stream1.block4,
        # model.backbone_stream1.block5,
        # model.backbone_stream1.block6,
        # model.backbone_stream1.block7,
        # model.backbone_stream1.block8,
        # model.backbone_stream1.block9,
        # model.backbone_stream1.block10,
        # model.backbone_stream1.block11,
        # model.backbone_stream1.block12,


        # stream2
        # model.backbone_stream2.block1,
        # model.backbone_stream2.block2,
        # model.backbone_stream2.block3,
        # model.backbone_stream2.block4,
        # model.backbone_stream2.block5,
        # model.backbone_stream2.block6,
        # model.backbone_stream2.block7,
        # model.backbone_stream2.block8,
        # model.backbone_stream2.block9,
        # model.backbone_stream2.block10,
        # model.backbone_stream2.block11,
        # model.backbone_stream2.block12,
        # model.L_attention,

        # fusion
        # model.crossfusion,

        #xception
        # model.backbone.block1,
        # model.backbone.block2,
        # model.backbone.block3,
        # model.backbone.block4,
        # model.backbone.block5,
        # model.backbone.block6,
        # model.backbone.block7,
        # model.backbone.block8,
        # model.backbone.block9,
        # model.backbone.block10,
        # model.backbone.block11,
        # model.backbone.block12,

        # mvd
        # model.backbone.base_model.model.encoder.layers[0],
        # model.backbone.base_model.model.encoder.layers[1],
        # model.backbone.base_model.model.encoder.layers[2],
        # model.backbone.base_model.model.encoder.layers[3],
        # model.backbone.base_model.model.encoder.layers[4],
        # model.backbone.base_model.model.encoder.layers[5],
        # model.backbone.base_model.model.encoder.layers[6],
        # model.backbone.base_model.model.encoder.layers[7],
        # model.backbone.base_model.model.encoder.layers[8],
        # model.backbone.base_model.model.encoder.layers[9],
        # model.backbone.base_model.model.encoder.layers[10],
        # model.backbone.base_model.model.encoder.layers[11],
        # model.backbone.base_model.model.encoder.layers[12],
        # model.backbone.base_model.model.encoder.layers[13],
        # model.backbone.base_model.model.encoder.layers[14],
        # model.backbone.base_model.model.encoder.layers[15],
        # model.backbone.base_model.model.encoder.layers[16],
        # model.backbone.base_model.model.encoder.layers[17],
        # model.backbone.base_model.model.encoder.layers[18],
        # model.backbone.base_model.model.encoder.layers[19],
        # model.backbone.base_model.model.encoder.layers[20],
        # model.backbone.base_model.model.encoder.layers[21],
        # model.backbone.base_model.model.encoder.layers[22],

        # 最后一层（通常保留这个即可，效果最直观）
        model.backbone.base_model.model.encoder.layers[23].layer_norm1,

    ]
    cam = GradCAM(model=model, target_layers=target_layers)

    labels_cpu = data_dict['label'].detach().cpu().tolist()
    targets = [ClassifierOutputTarget(int(y)) for y in labels_cpu]

    grayscale_cam = cam(data_dict=data_dict, targets=targets)  # 参考实现的核心调用
    # 统一转 numpy
    if isinstance(grayscale_cam, torch.Tensor):
        grayscale_cam = grayscale_cam.detach().cpu().numpy()
    elif isinstance(grayscale_cam, list):
        grayscale_cam = np.stack(grayscale_cam, axis=0)
    # 保证是 [B,H,W]
    if grayscale_cam.ndim == 2:
        grayscale_cam = grayscale_cam[None, ...]
    return grayscale_cam

# def show_cam_grid(original_rgb_uint8, cams, image_weight=0.6, mode='both'):
#     """
#     按参考实现风格拼图。
#     mode:
#       - 'both'    默认：第一行原图，第二行 叠加(JET)热力图
#       - 'orig'    只展示原图（单行）
#       - 'cam'     只展示伪彩热力图（单行）
#       - 'overlay' 只展示原图与热力图叠加图（单行）
#     """
#     B, H, W, _ = original_rgb_uint8.shape
#     rows = 2 if mode == 'both' else 1
#     fig, axs = plt.subplots(rows, B, figsize=(4.6 * B, 4.2 * rows), squeeze=False)
#
#     for i in range(B):
#         img_rgb = original_rgb_uint8[i]  # RGB uint8
#
#         # 如需热力图/叠加，则先准备对应图像
#         heatmap_rgb, overlay = None, None
#         if mode in ('cam', 'overlay', 'both'):
#             cam_i = cams[i]
#             # 尺寸对齐
#             if cam_i.shape[:2] != (H, W):
#                 cam_i = cv2.resize(cam_i, (W, H), interpolation=cv2.INTER_LINEAR)
#             # 归一化到 0-255
#             cmin, cmax = float(cam_i.min()), float(cam_i.max())
#             if cmax - cmin < 1e-6:
#                 cam_u8 = np.zeros((H, W), dtype=np.uint8)
#             else:
#                 cam_u8 = ((cam_i - cmin) / (cmax - cmin) * 255.0).astype(np.uint8)
#             # JET 伪彩
#             heatmap_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
#             heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
#             # 叠加
#             overlay = image_weight * img_rgb.astype(np.float32) + (1.0 - image_weight) * heatmap_rgb.astype(np.float32)
#             overlay = np.clip(overlay, 0, 255).astype(np.uint8)
#
#         # 绘制
#         if mode == 'orig':
#             axs[0, i].imshow(img_rgb)
#         elif mode == 'cam':
#             axs[0, i].imshow(heatmap_rgb)
#         elif mode == 'overlay':
#             axs[0, i].imshow(overlay)
#         elif mode == 'both':
#             axs[0, i].imshow(img_rgb)
#             axs[1, i].imshow(overlay)
#
#         for r in range(rows):
#             axs[r, i].axis('off')
#
#     plt.tight_layout()
#     plt.show()

def show_cam_grid(original_rgb_uint8, cams, image_weight=0.6, mode='overlay'):
    """
    按每三张为一组展示图片，组内无间隔，组间有明显空隙。
    mode: 支持 'orig' (仅显示原图) 或 'overlay' (仅显示原图与热力图叠加图)。
    """
    B, H, W, _ = original_rgb_uint8.shape
    rows = 1  # 只支持单行模式

    # 计算组数（每组3张）
    images_per_group = 3
    num_groups = (B + images_per_group - 1) // images_per_group  # 向上取整

    # 创建子图，增加组间列作为分隔
    total_cols = num_groups * images_per_group + max(0, num_groups - 1)  # 每组间加1列作为空隙
    fig = plt.figure(figsize=(3.5 * images_per_group * num_groups, 3.5))
    from matplotlib import gridspec

    # 设置宽度比例：图片列宽度1，空隙列宽度0.2（较小间距）
    width_ratios = []
    for g in range(num_groups):
        width_ratios.extend([1] * images_per_group)
        if g < num_groups - 1:
            width_ratios.append(0.2)  # 调整这个值来控制组间空隙大小（0.2 为更小间距）

    gs = gridspec.GridSpec(rows, total_cols, width_ratios=width_ratios, wspace=0.0, hspace=0.0)

    col_idx_accum = 0
    for i in range(B):
        img_rgb = original_rgb_uint8[i]  # RGB uint8

        # 准备热力图和叠加图（仅在 mode='overlay' 时使用）
        overlay = None
        if mode == 'overlay':
            cam_i = cams[i]
            if cam_i.shape[:2] != (H, W):
                cam_i = cv2.resize(cam_i, (W, H), interpolation=cv2.INTER_LINEAR)
            cmin, cmax = float(cam_i.min()), float(cam_i.max())
            if cmax - cmin < 1e-6:
                cam_u8 = np.zeros((H, W), dtype=np.uint8)
            else:
                cam_u8 = ((cam_i - cmin) / (cmax - cmin) * 255.0).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            overlay = image_weight * img_rgb.astype(np.float32) + (1.0 - image_weight) * heatmap_rgb.astype(np.float32)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # 计算当前图片的组索引和组内索引，添加组间空隙列
        group_idx = i // images_per_group
        within_group_idx = i % images_per_group
        col_offset = group_idx * (images_per_group + 1)  # 每组后加1列空隙
        col_idx = col_offset + within_group_idx

        # 绘制图片
        ax = fig.add_subplot(gs[0, col_idx])
        if mode == 'orig':
            ax.imshow(img_rgb)
        elif mode == 'overlay':
            ax.imshow(overlay)
        ax.axis('off')

    # 调整布局，确保组间空隙
    plt.tight_layout(pad=0.0)
    plt.show()

def main():
    # 1) 模型
    model = load_model()
    # 2) 数据 batch
    data_dict = build_data_dict(image_list, landmark_list, label_list)

    set_seed(1024)

    # 3) 生成 Grad-CAM（严格使用参考逻辑）
    cams = generate_gradcam(model, data_dict)  # [B,H,W]
    # 4) 反归一化得到原图（与参考实现一致：从模型输入反归一）
    imgs_rgb = _denorm_to_uint8(data_dict['image'], mean=0.5, std=0.5)  # [B,H,W,3] RGB uint8


    # 5) 拼图展示（第一行原图，第二行叠加热力图）
    show_cam_grid(imgs_rgb, cams, image_weight=0.6, mode='overlay')


if __name__ == "__main__":
    main()
