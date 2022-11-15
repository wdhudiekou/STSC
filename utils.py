import numpy as np
import torch
import time
# from skimage.measure import compare_psnr as psnr
import cv2
import os
import shutil
import math
from skimage.metrics import peak_signal_noise_ratio


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def PSNR_EDVR(img1, img2):
    '''
    img1 and img2 have range [0, 255]
    '''
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def psnr_tensor(label, predict, range):
    label = label.cpu().numpy()
    predict = predict.cpu().numpy()
    if label.ndim == 4:
        label = (label[0, :, :, :].transpose((1, 2, 0))+1) * 127.5 / range
        predict = (predict[0, :, :, :].transpose((1, 2, 0)) +1) * 127.5 / range
    label = np.clip(label, 0, 255)
    predict = np.clip(predict, 0, 255)
    return peak_signal_noise_ratio(label, predict)


def psnr_tensor_v2(label, predict, range):
    label = label.cpu().numpy()
    predict = predict.cpu().numpy()
    if label.ndim == 4:
        label = label[0, :, :, :].transpose((1, 2, 0)) * 255.0 / range
        predict = predict[0, :, :, :].transpose((1, 2, 0)) * 255.0 / range
    label = np.clip(label, 0, 255)
    predict = np.clip(predict, 0, 255)
    return peak_signal_noise_ratio(label, predict)



def measure_latency_in_ms(model, input_shape, is_cuda):
    INIT_TIMES = 50
    LAT_TIMES = 100
    lat = AverageMeter()
    model.eval()

    x = torch.randn(input_shape)
    if is_cuda:
        model = model.cuda()
        x = x.cuda()
    else:
        model = model.cpu()
        x = x.cpu()

    with torch.no_grad():
        for _ in range(INIT_TIMES):
            output = model(x)

        for _ in range(LAT_TIMES):
            torch.cuda.synchronize()
            tic = time.time()
            output = model(x)
            torch.cuda.synchronize()
            toc = time.time()
            lat.update(toc - tic, x.size(0))

    return lat.avg * 1000  # save as ms


def show_pair_data(label, predict, range=255):
    label = label.cpu().numpy()
    predict = predict.cpu().detach().numpy()
    if label.ndim == 4:
        label = label[0, :, :, :].transpose((1, 2, 0)) * 255.0 / range
        predict = predict[0, :, :, :].transpose((1, 2, 0)) * 255.0 / range
    label = np.clip(label, 0, 255)
    predict = np.clip(predict, 0, 255)
    label = label[:,:,::-1]
    predict = predict[:,:,::-1]
    cv2.imshow('label', label.astype(np.uint8))
    cv2.imshow('predict', predict.astype(np.uint8))
    cv2.waitKey(500)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def requires_grad(module, flag):
    for m in module.parameters():
        m.requires_grad = flag
