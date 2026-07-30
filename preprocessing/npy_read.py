import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from imutils import face_utils
import logging
import concurrent.futures


# 配置日志
def setup_logger():
    logger = logging.getLogger('landmark_extractor1')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # 文件输出
    log_path = Path.home() / 'landmark_extraction1.log'
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# 初始化dlib模型
def initialize_dlib_models():
    try:
        # 人脸检测器
        face_detector = dlib.get_frontal_face_detector()

        # 关键点预测器
        predictor_path = './dlib_tools/shape_predictor_81_face_landmarks.dat'
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Predictor file not found: {predictor_path}")

        face_predictor = dlib.shape_predictor(predictor_path)
        return face_detector, face_predictor

    except Exception as e:
        logger.error(f"Error initializing dlib models: {e}")
        raise


# 处理单张图片
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



# 处理单个视频帧目录
def process_video_frames(video_dir, frames_root, landmarks_root, face_detector, face_predictor):
    try:
        landmark_dir = landmarks_root / video_dir.name
        landmark_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(video_dir.glob('*.png'))  # 按顺序处理帧
        if not image_paths:
            logger.warning(f"No images found in: {video_dir}")
            return

        last_valid_landmarks = None  # 初始为空

        for img_path in image_paths:
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



# 主处理函数
def extract_landmarks_for_dfdc(frames_root_path):
    try:
        # 转换为Path对象
        frames_root = Path(frames_root_path)
        if not frames_root.exists() or not frames_root.is_dir():
            raise ValueError(f"Invalid frames directory: {frames_root_path}")

        # 创建landmarks根目录
        landmarks_root = frames_root.parent / "landmarks"
        landmarks_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Landmarks will be saved to: {landmarks_root}")

        # 初始化dlib模型
        face_detector, face_predictor = initialize_dlib_models()

        # 获取所有视频帧目录
        video_dirs = [d for d in frames_root.iterdir() if d.is_dir()]
        logger.info(f"Found {len(video_dirs)} video directories to process")

        # 使用多线程处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for video_dir in video_dirs:
                futures.append(
                    executor.submit(
                        process_video_frames,
                        video_dir,
                        frames_root,
                        landmarks_root,
                        face_detector,
                        face_predictor
                    )
                )

            # 等待所有任务完成
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(video_dirs),
                               desc="Processing videos"):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in processing: {e}")

        logger.info("Landmark extraction completed successfully!")

    except Exception as e:
        logger.error(f"Fatal error in landmark extraction: {e}")
        raise


if __name__ == "__main__":
    # 初始化日志
    logger = setup_logger()

    # 设置DFDC的frames路径
    dfdc_frames_path = "/home/csy/disk1/dataset/DeepfakeBench/rgb/DFDC/test/frames"

    # 执行处理
    extract_landmarks_for_dfdc(dfdc_frames_path)