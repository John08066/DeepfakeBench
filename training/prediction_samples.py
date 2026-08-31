"""
eval pretained model.
"""
import os
import sys
import random
import pickle

import pandas as pd
from matplotlib import pyplot as plt
from mmengine import DictAction, Config
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from collections import defaultdict
import numpy as np
import cv2
import torchvision.transforms as T
toPIL = T.ToPILImage()
import argparse
from logger import create_logger
from utils.grad_cam import GradCAM, ClassifierOutputTarget, show_cam_on_image

torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser(description='Deepfake Detection Test Args')
# parser.add_argument('--config_file', type=str,
#                     default='/home/jh/disk/workspace/DeepfakeBenchV2/training/config/detector/xception.yaml',
#                     help='path to detector YAML file')
parser.add_argument('--config_file', type=str,
                    default="/home/csy/disk1/project_1/deepfakeBench/DeepfakeBench-main/training/config/detector/aour.yaml",
                    help='path to detector YAML file')
parser.add_argument("--device_id", type=str, default='0')
# parser.add_argument('--checkpoints', type=str,
#                     default='/home/jh/disk/logs/DeepfakeBenchV2/xception/xception_FF_all_c23_20240702203635/test/Celeb-DF-v1/ckpt_best.pth')
parser.add_argument('--checkpoints', type=str,
                    default="/home/csy/disk1/project_1/deepfakeBench/DeepfakeBench-main/training/logs/training/aour1_2025-06-09-12-18-23/test/avg/ckpt_best.pth")
parser.add_argument('--grad_save_path', type=str,
                    default='/home/csy/disk1/project_1/deepfakeBench/DeepfakeBench-main/training/logs/training/aour1_2025-06-09-12-18-23/grad.pkl')

parser.add_argument("--opts", action=DictAction, help="Modify config options using the command-line", default=[],
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = 'customize'  # specify the current test dataset

        image_list = [
            '/home/jh/disk/datasets/deepfake/FaceForensics++/original_sequences/youtube/c23/frames/932/062.png',
            '/home/jh/disk/datasets/deepfake/FaceForensics++/original_sequences/youtube/c23/frames/932/124.png',
            '/home/jh/disk/datasets/deepfake/FaceForensics++/original_sequences/youtube/c23/frames/932/171.png',
            '/home/jh/disk/datasets/deepfake/FaceForensics++/original_sequences/youtube/c23/frames/932/186.png',
            '/home/jh/disk/datasets/deepfake/FaceForensics++/original_sequences/youtube/c23/frames/932/405.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/932_384/062.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/932_384/124.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/932_384/171.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/932_384/186.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/932_384/405.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/932_384/059.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/932_384/118.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/932_384/178.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/932_384/193.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/932_384/401.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/932_384/059.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/932_384/118.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/932_384/178.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/932_384/193.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/932_384/401.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/932_384/059.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/932_384/118.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/932_384/178.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/932_384/193.png',
            '/home/jh/disk1/datasets/deepfake/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/932_384/401.png',
        ]
        label_list = [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        spe_label_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        name_list = image_list
        test_set = DeepfakeAbstractBaseDataset(
            config=config,
            mode='predict',
            image_list=image_list,
            label_list=label_list,
            name_list=name_list,
            spe_label_list=spe_label_list,
        )

        # test_set.image_list =
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
            )
        return test_data_loader

    test_data_loader = get_test_data_loader(config, 'predict')
    return test_data_loader


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions

def grad_cam_one_dataset(model, data_loader, config):
    result = {'image':[], 'label':[], 'grad':[], 'mask':[]}
    for i, data_dict in enumerate(tqdm(data_loader)):
        # get data
        data, label, mask, landmark = \
            data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(
            device)
        data_dict['image'].requires_grad = True
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # grad_cam


        # ne_ms_fm_nlc_DA
        # target_layers = [
        #     model.backbone._blocks[1],
        #     model.backbone._blocks[5],
        #     model.backbone._blocks[9],
        #     model.backbone._blocks[15],
        #     model.backbone._blocks[21],
        #     model.backbone._blocks[29],
        #     model.backbone._blocks[31],
        #     model.backbone._conv_head
        # ]

        # rws
        target_layers = [
            model.backbone._blocks[1],
            model.backbone._blocks[5],
            model.backbone._blocks[9],
            model.backbone._blocks[15],
            model.backbone._blocks[21],
            model.backbone._blocks[29],
            model.backbone._blocks[31],
            model.backbone._conv_head
        ]

        # baseline
        # target_layers = [
        #     model.rgb_backbone._blocks[1],
        #     model.rgb_backbone._blocks[5],
        #     model.rgb_backbone._blocks[9],
        #     model.rgb_backbone._blocks[15],
        #     model.rgb_backbone._blocks[21],
        #     model.rgb_backbone._blocks[29],
        #     model.rgb_backbone._blocks[31],
        #     model.rgb_backbone._conv_head
        # ]

        # lsda
        # target_layers = [
        #     model.model.student_encoder.model._blocks[i] for i in range(31)
        # ]

        # xception
        # target_layers = [
        #     model.backbone.block1,
        #     model.backbone.block2,
        #     model.backbone.block3,
        #     model.backbone.block4,
        #     model.backbone.block5,
        #     model.backbone.block6,
        #     model.backbone.block7,
        #     model.backbone.block8,
        #     model.backbone.block9,
        #     model.backbone.block10,
        #     model.backbone.block11,
        #     model.backbone.block12,
        # ]

        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(i) for i in data_dict['label']]
        grayscale_cam = cam(data_dict=data_dict, targets=targets)

        data_dict['grad'] = grayscale_cam
        for i in result.keys():
            result[i].append(data_dict[i])

    for i in result.keys():
        temp = []
        for k in result[i]:
            if k is not None:
                for j in k:
                    temp.append(j)
        result[i] = temp
    return result

def predict_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    spe_label_list = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
            data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        # label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
    return np.array(prediction_lists), np.array(label_lists)


def predict_epoch(model, test_data_loader):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # result dict
    result_dict = defaultdict(list)

    # testing for all test data

    # compute loss for each dataset
    predictions_nps, label_list = predict_one_dataset(model, test_data_loader)

    result_dict['predictions_nps'].append(predictions_nps)
    result_dict['image_list'].append(test_data_loader.dataset.image_list)
    result_dict['label_list'].append(label_list)
    print('===> Predict Done!')
    return result_dict

def grad_cam(model, test_data_loader, config):
    # set model to eval mode
    model.eval()

    # testing for all test data
    data_dict = grad_cam_one_dataset(model, test_data_loader, config)
    return data_dict

def show_all_cam(data_dict,save_dir):
    # channel_mean = torch.tensor([0.485, 0.456, 0.406])
    # channel_std = torch.tensor([0.229, 0.224, 0.225])
    channel_mean = torch.tensor([0.5, 0.5, 0.5])
    channel_std = torch.tensor([0.5, 0.5, 0.5])
    # 这是反归一化的 mean 和std
    MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
    STD = [1 / std for std in channel_std]
    normalize = T.Normalize(mean=channel_mean, std=channel_std)
    denormalize = T.Normalize(mean=MEAN, std=STD)
    idx = 0

    def show_cam(idx):
        mean_grad = torch.Tensor(data_dict['grad'][idx])
        denormalized_img = denormalize(torch.Tensor(data_dict['image'][idx]))
        cam = np.array(toPIL(mean_grad))
        img = np.array(toPIL(denormalized_img))
        image_weight = 0.6
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        fusion = (1 - image_weight) * heatmap + image_weight * img
        fusion = fusion / np.max(fusion)
        return img, fusion

    fig, asex = plt.subplots(2, len(data_dict['image']), figsize=(73, 30))
    fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
    for j in range(len(data_dict['image'])):
        img, fusion = show_cam(j)
        asex[0][j].imshow(img)
        asex[1][j].imshow(fusion)
        asex[0][j].axis('off')  # 关闭坐标轴
        asex[1][j].axis('off')  # 关闭坐标轴
    plt.savefig(save_dir)

def main():
    # parse options and load config
    config = OmegaConf.load(args.config_file)
    config_test = OmegaConf.load('config/predict_config.yaml')
    config = OmegaConf.merge(config, config_test)
    config_args = OmegaConf.from_dotlist(args.opts)
    config = OmegaConf.merge(config, config_args)

    if args.checkpoints:
        checkpoints = args.checkpoints
        config['checkpoints'] = args.checkpoints
    else:
        checkpoints = config['checkpoints']

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loader = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model']['name']]
    model = model_class(config).to(device)
    get_parameter_number(model)
    if config['checkpoints']:
        ckpt = torch.load(config['checkpoints'], map_location=device)
        model.load_state_dict(ckpt, strict=True)

        print('===> Load checkpoint done!')
    else:
        print('Fail to load the checkpoint')
        sys.exit()

    # start testing
    result_dict = predict_epoch(model, test_data_loader)
    print('===> Predict Done!')

    # save tsne
    with open(os.path.join(config['log_dir'], f"predict_dict_{config['model']['name']}_{epoch}.pkl"), 'wb') as f:
        pickle.dump(result_dict, f)
    print('===> Save Prediction done!')

    import itertools
    import pandas as pd
    image_list = list(itertools.chain(*result_dict['image_list']))
    label_list = list(itertools.chain(*result_dict['label_list']))
    prediction_list = result_dict['predictions_nps']
    result = pd.DataFrame({'img_name': image_list, 'y_pred': np.stack(*prediction_list, axis=0), 'y':label_list})
    result.to_csv(os.path.join(config['log_dir'], 'prediction_test.csv'), index=None)
    print(result)

    # show Grad-CAM
    data_dict = grad_cam(model, test_data_loader, config)
    torch.save(data_dict, os.path.join(config['log_dir'], 'grad.pkl'))
    show_all_cam(data_dict, os.path.join(config['log_dir'], 'grad.png'))


if __name__ == '__main__':
    main()
