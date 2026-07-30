import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from imutils import face_utils
import logging


# 配置日志 (保持不变)
def setup_logger():
    logger = logging.getLogger('landmark_extractor')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log_path = Path.home() / 'landmark_extraction.log'
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# 初始化dlib模型 (保持不变)
def initialize_dlib_models():
    try:
        face_detector = dlib.get_frontal_face_detector()
        # 请确保此路径下有模型文件
        predictor_path = './dlib_tools/shape_predictor_81_face_landmarks.dat'
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Predictor file not found: {predictor_path}")
        face_predictor = dlib.shape_predictor(predictor_path)
        return face_detector, face_predictor
    except Exception as e:
        logger.error(f"Error initializing dlib models: {e}")
        raise


# 处理单张图片 (保持不变)
def process_image(img_path, face_detector, face_predictor, fallback_landmarks=None):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to read image: {img_path}")
            return fallback_landmarks

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_detector(rgb, 1)

        if not faces:
            logger.warning(f"No faces detected in: {img_path}")
            return fallback_landmarks

        face = max(faces, key=lambda rect: rect.width() * rect.height())
        shape = face_predictor(rgb, face)
        landmarks = face_utils.shape_to_np(shape)

        return landmarks.astype(np.uint32)

    except Exception as e:
        logger.error(f"Error processing image {img_path}: {e}")
        return fallback_landmarks


# 处理视频文件夹 (保持不变)
def process_video_frames(video_dir, landmarks_root, face_detector, face_predictor):
    try:
        # 核心逻辑：在 output_root 下创建同名文件夹
        # 例如: .../landmarks/video_01
        landmark_dir = landmarks_root / video_dir.name
        landmark_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(list(video_dir.glob('*.png')))  # 加上sorted保证顺序
        if not image_paths:
            logger.warning(f"No images found in: {video_dir}")
            return

        last_valid_landmarks = None

        for img_path in image_paths:
            # 核心逻辑：保持同名，后缀改为 .npy
            landmark_path = landmark_dir / f"{img_path.stem}.npy"
            if landmark_path.exists():
                continue

            landmarks = process_image(img_path, face_detector, face_predictor, last_valid_landmarks)

            if landmarks is not None:
                last_valid_landmarks = landmarks
                np.save(str(landmark_path), landmarks)
            else:
                logger.warning(f"Using fallback landmarks failed or none available: {img_path}")

    except Exception as e:
        logger.error(f"Error processing video directory {video_dir}: {e}")


# 【主要修改位置】增加了 output_root_path 参数
def extract_landmarks_for_dfdc(frames_root_path, output_root_path):
    try:
        frames_root = Path(frames_root_path)
        landmarks_root = Path(output_root_path)  # 直接使用传入的输出路径

        if not frames_root.exists() or not frames_root.is_dir():
            raise ValueError(f"Invalid frames directory: {frames_root_path}")

        # 创建输出根目录
        landmarks_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Input dir: {frames_root}")
        logger.info(f"Output dir: {landmarks_root}")

        face_detector, face_predictor = initialize_dlib_models()

        # 获取输入目录下的所有子文件夹（即视频文件夹）
        video_dirs = [d for d in frames_root.iterdir() if d.is_dir()]
        logger.info(f"Found {len(video_dirs)} video directories to process")

        for video_dir in tqdm(video_dirs, desc="Processing videos"):
            # 注意：process_video_frames 的调用签名稍微简化了，去掉了无用的 frames_root 参数
            process_video_frames(video_dir, landmarks_root, face_detector, face_predictor)

        logger.info("Landmark extraction completed successfully!")

    except Exception as e:
        logger.error(f"Fatal error in landmark extraction: {e}")
        raise


if __name__ == "__main__":
    logger = setup_logger()

    # 【配置输入路径】
    input_frames_path = "/datasets2/Deepfake/DeepfakeBench/rgb/DFDC/test/frames"

    # 【配置输出路径】
    output_landmarks_path = "/root/csy-7pw03c/disk/datasets/DFDC/landmarks"

    logger.info("Starting extraction task...")
    extract_landmarks_for_dfdc(input_frames_path, output_landmarks_path)