from torch.utils.data import DataLoader
import numpy as np
import torch
import cv2
from utils import show_pair_data
import torchvision.transforms as transforms


dataset_type = 'underwater_test'



if __name__ == '__main__':
    if dataset_type == 'underwater_train':
        from data.underwater_dataset import DerainTrainData
        data_dir = '/media/mark/新加卷/Dataset/Underwarter/UIEB640_480/'
        img_list = './list/rain_train.txt'
        dataset = DerainTrainData(data_dir, img_list, crop_size=(224,224))

    elif dataset_type == 'underwater_test':
        from data.underwater_dataset import DerainTestData
        data_dir = '/media/mark/新加卷/Dataset/Underwarter/UIEB640_480/'
        img_list = './list/rain_train.txt'
        dataset = DerainTestData(data_dir, img_list)
    else:
        pass
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)



    if dataset_type == 'underwater_train':
        for i, (noise, label) in enumerate(dataloader):
            show_pair_data(label, noise, 1)
    elif dataset_type == 'underwater_test':
        for i, (noise, label) in enumerate(dataloader):
            show_pair_data(label, noise, 1)
    else:
        for i, (noise, label,  _) in enumerate(dataloader):
            show_pair_data(noise, label, 1)


