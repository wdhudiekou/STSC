import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import vgg16
import utils

class MyVGG(nn.Module):
    def __init__(self):
        super(MyVGG, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.down = nn.AvgPool2d(2, stride=2)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        fea = []
        x_1 = self.act(self.conv1_1(x))
        x_1 = self.act(self.conv1_2(x_1))
        # (b, 64, 224, 224)

        x_1_down = self.down(x_1)

        x_2 = self.act(self.conv2_1(x_1_down))
        x_2 = self.act(self.conv2_2(x_2))
        # (b, 128, 112, 112)

        x_2_down = self.down(x_2)

        x_3 = self.act(self.conv3_1(x_2_down))
        x_3 = self.act(self.conv3_2(x_3))
        x_3 = self.act(self.conv3_3(x_3))
        # (b, 256, 56, 56)

        x_3_down = self.down(x_3)

        x_4 = self.act(self.conv4_1(x_3_down))
        x_4 = self.act(self.conv4_2(x_4))
        x_4 = self.act(self.conv4_3(x_4))
        # (b, 512, 28, 28)

        # x_4_down = self.down(x_4)
        #
        # x_5 = self.act(self.conv5_1(x_4_down))
        # x_5 = self.act(self.conv5_2(x_5))
        # x_5 = self.act(self.conv5_3(x_5))
        # (b, 512, 14, 14)

        # x_5_down = self.down(x_5)

        fea.append(x_1)
        fea.append(x_2)
        fea.append(x_3)
        fea.append(x_4)
        # fea.append(x_5)

        return fea


class VGGFeature(nn.Module):
    def __init__(self, normalize=True):
        super(VGGFeature, self).__init__()


        self.normalize = normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        self.vgg = MyVGG()

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        out = self.vgg(x)
        return out

if __name__ == '__main__':
    n = vgg16()
    print(n)
    # vgg_net = VGGFeature()
    # vgg_net.vgg.load_state_dict(torch.load('./MyVGG.pt'))



