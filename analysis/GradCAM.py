import os
import sys
import argparse
import yaml
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Import the AourDetector model and GradCAM utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.detectors.aour_detector import AourDetector
from utils.grad_cam import GradCAM, ClassifierOutputTarget, show_cam_on_image

# ======== Image & Landmark Loading ========
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.4, 0.5, 0.5])
])

def load_image(image_path, device):
    img = Image.open(image_path).convert('RGB')
    raw = img.copy().resize((256, 256))
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor, np.array(raw) / 255.0


def load_landmark(landmark_path, device):
    lm = np.load(landmark_path).astype(np.float32)
    return torch.from_numpy(lm).unsqueeze(0).to(device)

# ======== Model Loading ========

def load_model(config_path, weights_path, device):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    model = AourDetector(cfg).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# ======== Grad-CAM Generation ========

def generate_gradcam(model, image_tensor, landmark_tensor, target_layer):
    # Prepare GradCAM
    cam_extractor = GradCAM(model=model, target_layers=[target_layer])

    # Forward pass
    data_dict = {'image': image_tensor, 'landmark': landmark_tensor}
    # Use raw model forward without inference flag to allow hooks
    logits = model(data_dict, inference=True)['cls']
    class_idx = torch.argmax(logits, dim=1).item()

    # Compute CAM
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam_extractor(data_dict=data_dict, targets=targets)[0]

    # Normalize and colorize
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    return heatmap, class_idx

# ======== Visualization ========

def visualize_and_save(raw_image, heatmap, class_idx, save_path):
    overlay = heatmap * 0.4 + raw_image * 0.6
    overlay = overlay / overlay.max()

    plt.imshow(overlay)
    plt.title(f"Grad-CAM Overlay (Class {class_idx})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Grad-CAM saved to {save_path}")

# ======== Main ========

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM with Custom Loader for AourDetector")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--landmark', type=str, required=True, help='Path to landmark .npy file')
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights .pth')
    parser.add_argument('--output', type=str, default='./gradcam_output.png', help='Path to save Grad-CAM')
    parser.add_argument('--device', type=str, default='cuda:0', help='Computation device')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load data and model
    img_tensor, raw_img = load_image(args.image, device)
    landmark_tensor = load_landmark(args.landmark, device)
    model = load_model(args.config, args.weights, device)

    # Generate and visualize Grad-CAM
    heatmap, cls = generate_gradcam(model, img_tensor, landmark_tensor, model.crossfusion)
    visualize_and_save(raw_img, heatmap, cls, args.output)

if __name__ == '__main__':
    main()
