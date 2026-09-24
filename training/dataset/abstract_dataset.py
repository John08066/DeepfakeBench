# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

import sys

import lmdb



import os
import math
import yaml
import glob
import json

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T

import albumentations as A

from .albu import IsotropicResize
from path_config import resolve_rgb_path

# from utils.distortions import get_distortion_parameter, get_distortion_function
sys.path.append('.')
FFpp_pool = ['FaceForensics++', 'FaceShifter', 'DeepFakeDetection', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']  #


def all_in_pool(inputs, pool):
    for each in inputs:
        if each not in pool:
            return False
    return True


class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """

    def __init__(self, config=None, mode='train'):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """

        # Set the configuration and mode
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]

        # Check if 'video_mode' exists in config, otherwise set video_level to False
        self.video_level = config.get('video_mode', False)
        self.clip_size = config.get('clip_size', None)
        self.lmdb = config.get('lmdb', False)
        # Dataset dictionary
        self.image_list = []
        self.label_list = []
        self.spe_label_list = []

        # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = config['train_dataset']
            # Training data should be collected together for training
            image_list, label_list, spe_label_list = [], [], []
            for one_data in dataset_list:
                tmp_image, tmp_label, tmp_name, tmp_spe_label = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
                spe_label_list.extend(tmp_spe_label)
            if self.lmdb:
                if len(dataset_list) > 1:
                    if all_in_pool(dataset_list, FFpp_pool):
                        lmdb_path = os.path.join(config['lmdb_dir'], f"FaceForensics++_lmdb")
                        self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                    else:
                        raise ValueError('Training with multiple dataset and lmdb is not implemented yet.')
                else:
                    lmdb_path = os.path.join(config['lmdb_dir'],
                                             f"{dataset_list[0] if dataset_list[0] not in FFpp_pool else 'FaceForensics++'}_lmdb")
                    self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
        elif mode == 'test':
            one_data = config['test_dataset']
            # Test dataset should be evaluated separately. So collect only one dataset each time
            image_list, label_list, name_list, spe_label_list = self.collect_img_and_label_for_one_dataset(one_data)
            if self.lmdb:
                lmdb_path = os.path.join(config['lmdb_dir'],
                                         f"{one_data}_lmdb" if one_data not in FFpp_pool else 'FaceForensics++_lmdb')
                self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list) != 0 and len(label_list) != 0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list, self.spe_label_list = image_list, label_list, spe_label_list

        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list,
            'label': self.label_list,
            'spe_label': self.spe_label_list
        }

        self.transform = self.init_data_aug_method()

        # region  ==== 手动开关：测试期鲁棒性失真（默认 None 关闭）====
        # 读取扰动配置（类型和等级），若未配置则为 None / True
        # self.test_distortion = None
        # if self.test_distortion is not None:
        #     self.distortion_type = 'JPEG'     # 可选: CS | CC | BW | GNC | GB | JPEG
        #     self.distortion_level = 5  # 1 2 3 4 5
        # else:
        #     self.distortion_type = None
        #     self.distortion_level = None
        # endregion  ==== 手动开关：测试期鲁棒性失真（默认 None 关闭）====
        
    def init_data_aug_method(self):
        trans = A.Compose([
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR,
                                interpolation_up=cv2.INTER_LINEAR),
            ], p=0 if self.config['with_landmark'] else 1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'],
                                           contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'],
                               quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=True) if self.config[
                'with_landmark'] else None
        )
        return trans

    def rescale_landmarks(self, landmarks, original_size=256, new_size=224):
        scale_factor = new_size / original_size
        rescaled_landmarks = landmarks * scale_factor
        rescaled_landmarks = np.clip(rescaled_landmarks, 0, new_size - 1e-6)  # 避免浮点误差
        return rescaled_landmarks

    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """Collects image and label lists.

        Args:
            dataset_name (str): A list containing one dataset information. e.g., 'FF-F2F'

        Returns:
            list: A list of image paths.
            list: A list of labels.
            list: A list of video names.
            list: A list of specific labels (0=real, 1=type1, 2=type2...). # <--- 修改

        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
        # Initialize the label and frame path lists
        label_list = []
        frame_path_list = []

        # Record video name for video-level metrics
        video_name_list = []

        # <--- 新增: 初始化 "种类标签" 列表 ---
        spe_label_list = []

        # If JSON file exists, do the following data collection
        # FIXME: ugly, need to be modified here.
        cp = None
        if dataset_name == 'FaceForensics++_c40':
            dataset_name = 'FaceForensics++'
            cp = 'c40'
        elif dataset_name == 'FF-DF_c40':
            dataset_name = 'FF-DF'
            cp = 'c40'
        elif dataset_name == 'FF-F2F_c40':
            dataset_name = 'FF-F2F'
            cp = 'c40'
        elif dataset_name == 'FF-FS_c40':
            dataset_name = 'FF-FS'
            cp = 'c40'
        elif dataset_name == 'FF-NT_c40':
            dataset_name = 'FF-NT'
            cp = 'c40'

        # Try to get the dataset information from the JSON file
        if not os.path.exists(self.config['dataset_json_folder']):
            self.config['dataset_json_folder'] = self.config['dataset_json_folder'].replace(
                '/Youtu_Pangu_Security_Public', '/Youtu_Pangu_Security/public')
        try:
            with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f'dataset {dataset_name} not exist!')

        # Get the information for the current dataset
        for label_str in dataset_info[dataset_name]:  # (将 'label' 变量重命名为 'label_str' 避免歧义)
            sub_dataset_info = dataset_info[dataset_name][label_str][self.mode]
            # Special case for FaceForensics++ and DeepFakeDetection, choose the compression type
            if cp == None and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++',
                                               'DeepFakeDetection', 'FaceShifter']:
                sub_dataset_info = sub_dataset_info[self.compression]
            elif cp == 'c40' and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++',
                                                  'DeepFakeDetection', 'FaceShifter']:
                # 补全 c40 版本（从 c23 构造）
                if 'c23' in sub_dataset_info and 'c40' not in sub_dataset_info:
                    sub_dataset_info['c40'] = {}
                    for video_id, video_info in sub_dataset_info['c23'].items():
                        new_video_info = {'label': video_info['label']}
                        new_video_info['frames'] = [fp.replace('c23', 'c40') for fp in video_info['frames']]
                        sub_dataset_info['c40'][video_id] = new_video_info

                sub_dataset_info = sub_dataset_info['c40']

            # Iterate over the videos in the dataset
            for video_name, video_info in sub_dataset_info.items():
                # Unique video name
                unique_video_name = video_info['label'] + '_' + video_name

                # --- <--- 修改区域：同时获取两种标签 ---

                # 1. 获取二分类标签 (e.g., 0 or 1)
                video_label_string = video_info['label']  # (e.g., 'real', 'DF', 'F2F')
                if video_label_string not in self.config['label_dict']:
                    raise ValueError(f'Label {video_label_string} is not found in the configuration file.')
                label = self.config['label_dict'][video_label_string]

                # 2. 获取 "种类" 标签 (e.g., 0, 1, 2, 3, 4)
                if 'spe_label_dict' not in self.config:
                    raise ValueError('spe_label_dict is not defined in the configuration file.')
                if video_label_string not in self.config['spe_label_dict']:
                    raise ValueError(f'Label {video_label_string} is not found in the spe_label_dict.')
                spe_label = self.config['spe_label_dict'][video_label_string]

                # --- 结束修改 ---

                frame_paths = video_info['frames']
                if not frame_paths:  # 判断是否为空
                    print(f"Skipping video {unique_video_name} because frame_paths is empty.")
                    continue
                # sorted video path to the lists
                if '\\' in frame_paths[0]:
                    frame_paths = sorted(frame_paths, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
                else:
                    # frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                    frame_paths = sorted(
                        frame_paths,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]) if x.split('/')[-1].split('.')[0].isdigit()
                        else x.split('/')[-1].split('.')[0]
                    )

                total_frames = len(frame_paths)
                if self.frame_num < total_frames:
                    total_frames = self.frame_num
                    if self.video_level:
                        start_frame = random.randint(0, total_frames - self.frame_num) if self.mode == 'train' else 0
                        frame_paths = frame_paths[start_frame:start_frame + self.frame_num]  # update total_frames
                    else:
                        step = total_frames // self.frame_num
                        frame_paths = [frame_paths[i] for i in range(0, total_frames, step)][:self.frame_num]

                # If video-level methods, crop clips from the selected frames if needed
                if self.video_level:
                    if self.clip_size is None:
                        raise ValueError('clip_size must be specified when video_level is True.')
                    if total_frames >= self.clip_size:
                        selected_clips = []
                        num_clips = total_frames // self.clip_size
                        if num_clips > 1:
                            clip_step = (total_frames - self.clip_size) // (num_clips - 1)
                            for i in range(num_clips):
                                start_frame = random.randrange(i * clip_step, min((i + 1) * clip_step,
                                                                                  total_frames - self.clip_size + 1)) if self.mode == 'train' else i * clip_step
                                continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                                assert len(
                                    continuous_frames) == self.clip_size, 'clip_size is not equal to the length of frame_path_list'
                                selected_clips.append(continuous_frames)
                        else:
                            start_frame = random.randrange(0,
                                                           total_frames - self.clip_size + 1) if self.mode == 'train' else 0
                            continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                            assert len(
                                continuous_frames) == self.clip_size, 'clip_size is not equal to the length of frame_path_list'
                            selected_clips.append(continuous_frames)

                        # --- <--- 修改区域：添加 spe_label ---
                        # Append the list of selected clips and append the label
                        label_list.extend([label] * len(selected_clips))
                        spe_label_list.extend([spe_label] * len(selected_clips))  # <--- 新增
                        frame_path_list.extend(selected_clips)
                        # video name save
                        video_name_list.extend([unique_video_name] * len(selected_clips))

                    else:
                        print(
                            f"Skipping video {unique_video_name} because it has less than clip_size ({self.clip_size}) frames ({total_frames}).")

                # Otherwise, extend the label and frame paths to the lists according to the number of frames
                else:
                    # --- <--- 修改区域：添加 spe_label ---
                    # Extend the label and frame paths to the lists according to the number of frames
                    label_list.extend([label] * total_frames)
                    spe_label_list.extend([spe_label] * total_frames)  # <--- 新增
                    frame_path_list.extend(frame_paths)
                    # video name save
                    video_name_list.extend([unique_video_name] * len(frame_paths))

        # --- <--- 修改区域：打包和返回 spe_label ---

        # Shuffle the label and frame path lists in the same order
        shuffled = list(zip(label_list, frame_path_list, video_name_list, spe_label_list))  # <--- 修改
        random.shuffle(shuffled)
        label_list, frame_path_list, video_name_list, spe_label_list = zip(*shuffled)  # <--- 修改

        return frame_path_list, label_list, video_name_list, spe_label_list  # <--- 修改

    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution']  # if self.mode == "train" else self.config['resolution']
        if not self.lmdb:
            file_path = resolve_rgb_path(file_path, self.config['rgb_dir'])

            assert os.path.exists(file_path), f"{file_path} does not exist"
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError('Loaded image is None: {}'.format(file_path))
        elif self.lmdb:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0] == '.':
                    file_path = file_path.replace('./datasets\\', '')

                image_bin = txn.get(file_path.encode())
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1))
        if not self.lmdb:
            file_path = resolve_rgb_path(file_path, self.config['rgb_dir'])
            if os.path.exists(file_path):
                mask = cv2.imread(file_path, 0)
                if mask is None:
                    mask = np.zeros((size, size))
            else:
                return np.zeros((size, size, 1))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0] == '.':
                    file_path = file_path.replace('./datasets\\', '')

                image_bin = txn.get(file_path.encode())
                if image_bin is None:
                    mask = np.zeros((size, size, 3))
                else:
                    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                    # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
                    mask = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (size, size)) / 255
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))

        if not self.lmdb:
            file_path = resolve_rgb_path(file_path, self.config['rgb_dir'])
            if os.path.exists(file_path):
                landmark = np.load(file_path)
                landmark = self.rescale_landmarks(np.float32(landmark), original_size=256,
                                                  new_size=self.config['resolution'])
            else:
                return np.zeros((81, 2))

        # # 下面是为了单独处理dfdc
        # elif file_path.startswith('DFDC'):
        #     file_path = f'{self.config["rgb_dir"]}/' + file_path.replace('\\', '/')
        #     landmark = np.load(file_path)
        #     landmark = self.rescale_landmarks(np.float32(landmark), original_size=256,
        #                                           new_size=self.config['resolution'])

        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0] == '.':
                    file_path = file_path.replace('./datasets\\', '')
                binary = txn.get(file_path.encode())
                landmark = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
                landmark = self.rescale_landmarks(np.float32(landmark), original_size=256,
                                                  new_size=self.config['resolution'])
        return landmark

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Set the seed for the random number generator
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)

        # Create a dictionary of arguments
        kwargs = {'image': img}

        # Check if the landmark and mask are not None
        if landmark is not None:
            h, w = img.shape[:2]
            landmark[:, 0] = np.clip(landmark[:, 0], 0, w - 1e-3)  # x
            landmark[:, 1] = np.clip(landmark[:, 1], 0, h - 1e-3)  # y
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy', remove_invisible=True)
        if mask is not None:
            mask = mask.squeeze(2)
            if mask.max() > 0:
                kwargs['mask'] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)

        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask', mask)

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        # Reset the seeds to ensure different transformations for different videos
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index, no_norm=False):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_paths = self.data_dict['image'][index]
        # image_paths = f'{self.config["rgb_dir"]}/' + image_paths.replace('\\', '/')  # 新加的
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            image_paths = [image_paths]  # for the image-level IO, only one frame is used

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            # Initialize a new seed for data augmentation at the start of each video
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2 ** 32 - 1)

            # Get the mask and landmark paths
            # 这里加个判断如果是df40的话就将ff换为landmarks
            if image_path.startswith('DF40'):
                landmark_path = image_path.replace('ff', 'landmarks_ff').replace('.png',
                                                                                 '.npy')  # Use .npy for landmark
            else:
                landmark_path = image_path.replace('frames', 'landmarks').replace('.png',
                                                                                  '.npy')  # Use .npy for landmark

            mask_path = image_path.replace('frames', 'masks')  # Use .png for mask

            # Load the image
            try:
                image = self.load_rgb(image_path)
            except Exception as e:
                # Skip this image and return the first one
                print(f"Error loading image at index {index}: {e}")
                return self.__getitem__(0)
            image = np.array(image)  # Convert to numpy array for data augmentation

            # Load mask and landmark (if needed)
            if self.config['with_mask']:
                mask = self.load_mask(mask_path)
            else:
                mask = None
            if self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
            else:
                landmarks = None

            # Do Data Augmentation
            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask, augmentation_seed)
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)

            # ---- 测试期鲁棒性失真：只作用于图像（光度变换），不改 landmarks/mask ----
            # if self.mode == 'test' and self.distortion_type is not None:
            #     # 获取参数并调用对应的扰动函数
            #     img_bgr = cv2.cvtColor(image_trans, cv2.COLOR_RGB2BGR)
            #     param = get_distortion_parameter(self.distortion_type, self.distortion_level)
            #     func = get_distortion_function(self.distortion_type)
            #     img_bgr = func(img_bgr, param)
            #     image_trans = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # ------------------------------------------------------------------------

            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                if self.config['with_landmark']:
                    landmarks_trans = torch.from_numpy(landmarks)
                if self.config['with_mask']:
                    mask_trans = torch.from_numpy(mask_trans)

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        if self.video_level:
            # Stack image tensors along a new dimension (time)
            image_tensors = torch.stack(image_tensors, dim=0)
            # Stack landmark and mask tensors along a new dimension (time)
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in
                       landmark_tensors):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
            # Get the first image tensor
            image_tensors = image_tensors[0]
            # Get the first landmark and mask tensors
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in
                       landmark_tensors):
                landmark_tensors = landmark_tensors[0]
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = mask_tensors[0]

        return image_tensors, label, landmark_tensors, mask_tensors

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, labels, landmarks, masks = zip(*batch)

        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)

        # Special case for landmarks and masks if they are None
        if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmarks):
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if not any(m is None or (isinstance(m, list) and None in m) for m in masks):
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)


if __name__ == "__main__":
    with open('/data/home/zhiyuanyan/DeepfakeBench/training/config/detector/video_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = DeepfakeAbstractBaseDataset(
        config=config,
        mode='train',
    )
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True,
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm

    for iteration, batch in enumerate(tqdm(train_data_loader)):
        # print(iteration)
        ...
        # if iteration > 10:
        #     break
