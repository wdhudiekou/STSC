from PIL import Image
import torch.utils.data as data
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
import re
import os
import numpy as np

class DerainTrainData(data.Dataset):
    def __init__(self, data_dir, name_file, crop_size=224):
        super().__init__()

        img_list = name_file

        with open(img_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]

        self.input_names = input_names
        self.gt_names = input_names        # same name
        self.crop_size = crop_size
        self.gt_dir = data_dir + '/trainB/'
        self.input_dir = data_dir + '/trainA/'


    def crop_image(self, input, label):
        crop_width, crop_height = self.crop_size
        width, height = input.size
        if width < crop_width and height < crop_height :
            input = input.resize((crop_width,crop_height), Image.ANTIALIAS)
            label = label.resize((crop_width, crop_height), Image.ANTIALIAS)

        elif width < crop_width :
            input = input.resize((crop_width,height), Image.ANTIALIAS)
            label = label.resize((crop_width,height), Image.ANTIALIAS)

        elif height < crop_height :
            input = input.resize((width,crop_height), Image.ANTIALIAS)
            label = label.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input.size

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = label.crop((x, y, x + crop_width, y + crop_height))

        return input_crop_img, gt_crop_img

    def get_images(self, index):

        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        input_img = Image.open(self.input_dir + input_name)

        # # extra
        # img_np = np.asarray(input_img, np.float)
        # img_np /= 255
        #
        # img_max = np.max(img_np, axis=(0,1))
        # img_min = np.min(img_np, axis=(0,1))
        #
        # img_pre = (img_np - img_min) / (img_max-img_min)
        # input_img = Image.fromarray(np.uint8(img_pre*255))

        try:
            gt_img = Image.open(self.gt_dir + gt_name)
        except:
            gt_img = Image.open(self.gt_dir + gt_name).convert('RGB')

        ## crop data

        input_img, gt_img = self.crop_image(input_img, gt_img)

        #print(input_img.size)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor()])    # , Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #transform_input = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        input_t = transform_input(input_img)
        gt_t = transform_gt(gt_img)

        # --- Check the channel is 3 or not --- #
        if list(input_t.shape)[0] is not 3 or list(gt_t.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return input_t, gt_t


    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


class DerainTestData(data.Dataset):
    def __init__(self, data_dir, name_file):
        super().__init__()

        with open(name_file) as f:
            contents = f.readlines()
            gt_names = [i.strip() for i in contents]

        self.input_names = gt_names
        self.gt_names = gt_names
        self.gt_data_dir = data_dir + '/testB_1/'
        self.input_data_dir = data_dir + '/testA_1/'


    def crop_image(self, input, label):

        width, height = input.size
        width = width // 16 * 16
        height = height // 16 * 16
        input_crop_img = input.crop((0, 0, width, height)) # .resize((480,400), Image.ANTIALIAS)
        gt_crop_img = label.crop((0, 0, width, height)) # .resize((480,400), Image.ANTIALIAS)
        return input_crop_img, gt_crop_img


    def get_images(self, index):

        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

       #  image_id = input_name.split('_')[1].split('.')[0]

        input_img = Image.open(self.input_data_dir + input_name)

        try:
            gt_img = Image.open(self.gt_data_dir + gt_name)
        except:
            gt_img = Image.open(self.gt_data_dir + gt_name).convert('RGB')


        ## crop data
        input_img, gt_img = self.crop_image(input_img, gt_img)

        # print(input_img.size)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor()])    # , Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform_gt = Compose([ToTensor()])
        input_t = transform_input(input_img)
        gt_t = transform_gt(gt_img)

        # --- Check the channel is 3 or not --- #
        if list(input_t.shape)[0] is not 3 or list(gt_t.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        input_main_name = input_name.split('.')[0]
        return input_t, gt_t, input_main_name


    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)





class DerainTestData_noGT(data.Dataset):
    def __init__(self, data_dir, name_file):
        super().__init__()

        with open(name_file) as f:
            contents = f.readlines()
            gt_names = [i.strip() for i in contents]

        self.input_names = gt_names
        self.input_data_dir = data_dir + '/testA/'


    def crop_image(self, input):

        width, height = input.size
        width = width // 32 * 32
        height = height // 32 * 32
        input_crop_img = input.crop((0, 0, width, height))
        return input_crop_img


    def get_images(self, index):

        input_name = self.input_names[index]

       #  image_id = input_name.split('_')[1].split('.')[0]

        input_img = Image.open(self.input_data_dir + input_name)


        ## crop data
        input_img = self.crop_image(input_img)

        # print(input_img.size)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor()])    # , Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        input_t = transform_input(input_img)

        return input_t, input_name.split('.')[0]


    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


