import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import torch

def tensor_to_images(image_tensor, channel_mean, channel_std):

    MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
    STD = [1 / std for std in channel_std]
    denormalize = T.Normalize(mean=MEAN, std=STD)
    toPIL = T.ToPILImage()
    denormalized_img = denormalize(torch.Tensor(image_tensor))
    img = toPIL(denormalized_img)
    return img

def show_img(img):
    plt.imshow(img)
    plt.show()