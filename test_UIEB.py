import os
import sys
import torch
import logging
import argparse

import cv2
import numpy as np
from PIL import Image
from torch.utils import data

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from data.underwater_dataset import DerainTrainData, DerainTestData, DerainTestData_noGT
from model.my_vgg import VGGFeature
# from model.networks_v4_nores_new_concat import UnderNet
# from model.networks_v4_nores_new_st_te import UnderNet
# from model.networks_v4_nores_new_concat_decoder import UnderNet
from model.networks_v4_nores_new_st_te_decoder import UnderNet



def Tensor2image(img):
    img = img.data.cpu().numpy()
    img[img > 1] = 1
    img[img < 0] = 0
    img *= 255
    img = img.astype(np.uint8)[0]
    img = img.transpose((1, 2, 0))
    return img

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
            # feature = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
            save_path = os.path.join(map_dir, 'map_%s_%s.jpg' % (str(_batch), str(_channel)))
            print(save_path)
            cv2.imwrite(save_path, feature)

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='./checkpoint/underwater-20210918-155028/090000.pt')
    parser.add_argument("--iter", type=int, default=500000)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dim_z", type=int, default=128)
    parser.add_argument("--dim_class", type=int, default=128)
    parser.add_argument("--rec_weight", type=float, default=0.1)
    parser.add_argument("--div_weight", type=float, default=0.1)
    parser.add_argument("--crop_prob", type=float, default=0.3)
    parser.add_argument("--n_class", type=int, default=10)
    #parser.add_argument("path", metavar="PATH")
    parser.add_argument("--train_root", type=str, default='/home/zongzong/WD/Datasets/UnderWater/UIEB/UIEB/UIEB640')
    parser.add_argument("--train_list", type=str, default='./data/list/UIEB_test.txt')
    parser.add_argument('--save', type=str, default='./checkpoint/underwater-20210918-155028', help='method of test')
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument('--gpu', type=str, default='0', help='gpu_id')
    args = parser.parse_args()
    args.distributed = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'test_log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    dset       = DerainTestData(args.train_root, args.train_list)
    dataloader = data.DataLoader(dset, batch_size=1, shuffle=False)

    test_epoch = 99999

    show_dst = args.save + '/USOD_test_onlyvggfea:{}/'.format(test_epoch)
    if not os.path.exists(show_dst):
        os.makedirs(show_dst)

    vgg = VGGFeature()
    vgg.load_state_dict(torch.load('./pretrain/vggfeature.pth'))
    vgg = vgg.eval().cuda()

    gen = UnderNet(3, 3, ngf=32, weight=0.5).cuda()
    gen.load_state_dict(torch.load(args.save + '/{:06d}.pt'.format(test_epoch))['g'])
    gen = gen.eval().cuda()

    psnr_sum = []
    ssim_sum = []
    test_num = 0
    img_num = 0
    for i, (input_img, label, name) in enumerate(dataloader):

        with torch.no_grad():
            input_img = input_img.cuda()
            label_in = label.cuda()
            img_num += 1

            input_ = input_img
            fea_input = vgg(input_)
            out = gen(input_, fea_input)

            # TODO: running time
            # st = time.time()
            # fea_input = vgg(input_)
            # out = gen(input_, fea_input)
            # run_time = time.time() - st
            # print("Running Time: {:.3f}s\n".format(run_time))
            # exit(00)

        img = Tensor2image(out)
        gt  = Tensor2image(label_in)

        Image.fromarray(img).save(show_dst + name[0] + '.png')

        P = peak_signal_noise_ratio(img, gt)
        S = structural_similarity(img, gt, multichannel=True)

        psnr_sum.append(P)
        ssim_sum.append(S)

    print(np.mean(psnr_sum))
    print(np.mean(ssim_sum))
    logging.info('Test: %d', test_epoch)
    logging.info('Stage: %d   PSNR avg:  %f     SSIM avg:   %f', (i + 1), np.mean(psnr_sum), np.mean(ssim_sum))

