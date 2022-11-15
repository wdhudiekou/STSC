import argparse
import os
import numpy as np
import cv2
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
import torchvision
from torch.autograd import Variable
from data.underwater_dataset import DerainTrainData
from model.model import VGGFeature
# from model.networks_v4_nores_new_concat import UnderNet
# from model.networks_v4_nores_new_st_te import UnderNet
from model.networks_v4_nores_new_st_te_decoder import UnderNet
# from model.networks_v4_nores_new_concat_decoder import UnderNet
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import sys
import logging
import random

import time
import utils
import glob


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--iter", type=int, default=100000)
parser.add_argument("--start_iter", type=int, default=0)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--dim_z", type=int, default=128)
parser.add_argument("--dim_class", type=int, default=128)
parser.add_argument("--rec_weight", type=float, default=0.2)
parser.add_argument("--div_weight", type=float, default=0.1)
parser.add_argument("--crop_prob", type=float, default=0.3)
parser.add_argument("--n_class", type=int, default=10)
parser.add_argument("--train_root", type=str, default='/home/zongzong/WD/Datasets/UnderWater/UIEB/UIEB/UIEB640')
parser.add_argument("--train_list", type=str, default='./data/list/UIEB_train.txt')
parser.add_argument("--patch_size", type=int, default=224)
parser.add_argument('--gpu', type=str, default='0', help='gpu_id')
parser.add_argument('--resume', type=str, default='', help='gpu_id')
args = parser.parse_args()
args.distributed = False

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.resume:
    save = args.resume
    checkpoint_path = save
else:
    save = '{}-{}'.format('underwater', time.strftime("%Y%m%d-%H%M%S"))
    checkpoint_path = './checkpoint/' + save
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
utils.create_exp_dir(checkpoint_path, scripts_to_save=glob.glob('*.py') + glob.glob('./model/*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(checkpoint_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info('Run file:  %s', __file__)
logging.info('args= %s', args)



class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)



def patch_loss(net, criten, real_l, fake_l):
    loss = []
    for real, fake in zip(real_l, fake_l):
        real_pred = net(real)
        fake_pred = net(fake)
        loss.append(criten_loss(criten, real_pred, fake_pred, False))

    return sum(loss)/4


def criten_loss(criten, real_pred, fake_pred, use_lsgan=True):
    if use_lsgan:
        loss_D = (criten(real_pred - torch.mean(fake_pred), True) +
                  criten(fake_pred - torch.mean(real_pred), False)) / 2
    else:
        loss_D = (criten(real_pred, True) + criten(fake_pred, False))/2
    return loss_D


def requires_grad(module, flag):
    for m in module.parameters():
        m.requires_grad = flag


def d_ls_loss(real_predict, fake_predict):
    loss = (real_predict - 1).pow(2).mean() + fake_predict.pow(2).mean()

    return loss


def g_ls_loss(real_predict, fake_predict):
    loss = (fake_predict - 1).pow(2).mean()

    return loss


def recon_loss(features_fake, features_real):
    r_loss = []

    for f_fake, f_real in zip(features_fake, features_real):
        ll = (F.l1_loss(f_fake, f_real, reduction="none")).mean()
        r_loss.append(ll)

    return sum(r_loss)


def diversity_loss(z1, z2, fake1, fake2, eps=1e-8):
    div_z = F.l1_loss(z1, z2, reduction="none").mean(1)
    div_fake = F.l1_loss(fake1, fake2, reduction="none").mean((1, 2, 3))

    d_loss = (div_z / (div_fake + eps)).mean()
    return d_loss


def mse_loss(fake, real):
    mse = F.mse_loss(fake, real, reduction="none").mean()
    return mse


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)

            yield next(loader_iter)


def get_patch(real, fake, n, patch_size):
    w = real.size(3)
    h = real.size(2)
    real_patch_1 = []
    fake_patch_1 = []
    for i in range(n):
        w_offset_1 = random.randint(0, max(0, w - patch_size - 1))
        h_offset_1 = random.randint(0, max(0, h - patch_size - 1))
        real_patch_1.append(real[:, :, h_offset_1:h_offset_1 + patch_size,
                                 w_offset_1:w_offset_1 + patch_size])
        fake_patch_1.append(fake[:, :, h_offset_1:h_offset_1 + patch_size,
                                 w_offset_1:w_offset_1 + patch_size])
    return real_patch_1, fake_patch_1

def get_feature_map(tensor_data, map_dir):
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
    b, c, h, w = tensor_data.size()
    for _batch in range(b):
        for _channel in range(c):
            feature = tensor_data[_batch, _channel, :, :]
            feature  = feature.view(h, w)
            feature = feature.data.to('cpu').numpy()
            feature = 1.0 / (1 + np.exp(-1 * feature))
            feature = np.round(feature*255.0)
            save_path = os.path.join(map_dir, 'map_%s_%s.jpg' % (str(_batch), str(_channel)))
            print(save_path)
            cv2.imwrite(save_path, feature)


def train(args, dataset_pair, dataset_unpair, gen, dis_l=None, dis_g=None):

    vgg = VGGFeature(True)
    vgg.load_state_dict(torch.load('./pretrain/vggfeature.pth'))
    vgg = vgg.cuda().eval()
    requires_grad(vgg, False)# TODO: do not update the parameter of the VGG16

    g_optim = optim.Adam(gen.parameters(), lr=5e-4, betas=(0, 0.999))

    loader = data.DataLoader(
        dataset_pair,
        batch_size=args.batch,
        num_workers=3,
        shuffle=True,
        drop_last=True,
    )
    loader_iter = sample_data(loader)
    gen.train()

    L1_loss = nn.L1Loss()
    L1_loss = L1_loss.cuda()

    ssim_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True)
    ssim_loss = ssim_loss.cuda()

    pbar = range(args.start_iter, args.iter)
    eps = 1e-8

    for i in pbar:
        input_img, real = next(loader_iter)

        real      = real.cuda()
        input_img = input_img.cuda()

        features = vgg(input_img)
        g_optim.zero_grad()
        fake1   = gen(input_img, features)
        mseloss = L1_loss(fake1, real)
        _ssim_loss = 1 - ssim_loss(real, fake1)

        g_loss = args.rec_weight * mseloss + _ssim_loss * 0.8
        g_loss.backward()
        g_optim.step()

        if ((i+1) % 50 ==0):
            logging.info('step: %d   r_loss: %.4f  mse: %.4f', i, _ssim_loss.item(), mseloss.item())

        if ((i+1) % 500 ==0):
            torchvision.utils.save_image(
                fake1,
                f"sample/{str(i).zfill(6)}.png",
                nrow=int(args.batch ** 0.5),
                normalize=True,
                range=(0, 1),
            )
            torchvision.utils.save_image(
                input_img,
                f"sample/{str(i).zfill(6)}_input.png",
                nrow=int(args.batch ** 0.5),
                normalize=True,
                range=(0, 1),
            )
            torchvision.utils.save_image(
                real,
                f"sample/{str(i).zfill(6)}_real.png",
                nrow=int(args.batch ** 0.5),
                normalize=True,
                range=(0, 1),
            )

        if ((i+1) % 10000 == 0):
            torch.save(
                {
                    "args": args,
                    "g": gen.state_dict(),
                },
                "{}/{:06d}.pt".format(checkpoint_path, i),
            )
        if ((i+1)% 8000==0):
            current_lr = g_optim.param_groups[0]['lr']
            for param_group in g_optim.param_groups:
                param_group['lr'] = current_lr / 5


if __name__ == "__main__":

    dataset_pair = DerainTrainData(args.train_root, args.train_list, crop_size=(args.patch_size, args.patch_size))
    gen = UnderNet(3, 3, ngf=32, weight=0.5).cuda()
    train(args, dataset_pair, None,  gen, None, None)
