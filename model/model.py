import torch
import torch.nn as nn
from model.my_vgg import VGGFeature
import numpy as np
import torch.nn.functional as F


# class ResBlock(nn.Module):
#     def __init__(self, C_in, C_out, bn=False, act='relu', upsample=False, downsample=False):
#         super(ResBlock, self).__init__()
#
#         self.upsample = upsample
#         self.downsample = downsample
#         self.isrelu = act
#         self.isbn = bn
#
#
#         if upsample:
#             self.up = nn.Upsample(scale_factor=2)
#         if downsample:
#             self.down = nn.Conv2d(C_in, C_in, 3, 2, 1)
#
#         self.conv1 = nn.Conv2d(C_in, C_out, 3, 1, 1)
#
#         self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, 1)
#
#         if act == 'relu':
#             self.act = nn.ReLU(inplace=True)
#         elif act == 'lrelu':
#             self.act = nn.LeakyReLU(0.2)
#         elif act == 'prelu':
#             self.act1 = nn.PReLU()
#             self.act2 = nn.PReLU()
#
#         if bn:
#             self.bn1 = nn.BatchNorm2d(C_out)
#             self.bn2 = nn.BatchNorm2d(C_out)
#
#
#     def forward(self, x):
#         if self.upsample:
#             x = self.up(x)
#         if self.downsample:
#             x = self.down(x)
#
#         x = self.conv1(x)
#         if self.isrelu == 'prelu':
#             x = self.act1(x)
#         else:
#             x = self.act(x)
#
#         if self.isbn:
#             x = self.bn1(x)
#
#         x = self.conv2(x)
#         if self.isrelu == 'prelu':
#             x = self.act2(x)
#         else:
#             x = self.act(x)
#
#         if self.isbn:
#             x = self.bn2(x)
#
#         return x


class ResOp(nn.Module):
    def __init__(self, C_in, C_out):
        super(ResOp, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        if self.C_out != self.C_in:
            self.head = nn.Conv2d(C_in, C_out, 1, 1, 0)
        self.body = nn.Sequential(
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        if self.C_out != self.C_in:
           x = self.head(x)
        res = self.body(x)
        return res


class UNet(nn.Module):     ## V1 fusion
    def __init__(self):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 512]

        self.head = nn.Conv2d(3, nb_filter[0], 3, 1, 1)

        self.down1 = nn.Conv2d(nb_filter[0], nb_filter[1], 3, 2, 1)
        self.down2 = nn.Conv2d(nb_filter[1], nb_filter[2], 3, 2, 1)
        self.down3 = nn.Conv2d(nb_filter[2], nb_filter[3], 3, 2, 1)
        self.down4 = nn.Conv2d(nb_filter[3], nb_filter[4], 3, 2, 1)

        self.conv0_0 = ResOp(nb_filter[0], nb_filter[0])
        # self.conv0_1 = ResOp(nb_filter[0], nb_filter[0])
        self.conv1_0 = ResOp(nb_filter[1], nb_filter[1])
        # self.conv1_1 = ResOp(nb_filter[1], nb_filter[1])
        self.conv2_0 = ResOp(nb_filter[2], nb_filter[2])
        # self.conv2_1 = ResOp(nb_filter[2], nb_filter[2])
        self.conv3_0 = ResOp(nb_filter[3], nb_filter[3])
        #  self.conv3_1 = ResOp(nb_filter[3], nb_filter[3])
        self.conv4_0 = ResOp(nb_filter[4], nb_filter[4])

    def forward(self, input):

        fea = self.head(input)
        x0_0 = self.conv0_0(fea)
        x1_0 = self.conv1_0(self.down1(x0_0))
        x2_0 = self.conv2_0(self.down2(x1_0))
        x3_0 = self.conv3_0(self.down3(x2_0))
        x4_0 = self.conv4_0(self.down4(x3_0))

        return x4_0


class NoNormDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False):
        super(NoNormDiscriminator, self).__init__()

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)




def get_activation(name):
    if name == 'leaky_relu':
        activation = nn.LeakyReLU(0.2)

    elif name == 'relu':
        activation = nn.ReLU()

    return activation

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        norm=False,
        upsample=False,
        downsample=False,
        first=False,
        activation='relu',
    ):
        super().__init__()

        self.first = first
        self.norm = norm

        bias = False if norm else True

        if norm:
            self.norm1 = nn.BatchNorm2d(in_channel)

        if not self.first:
            self.activation1 = get_activation(activation)

        if upsample:
            self.upsample = nn.Upsample(scale_factor=2)

        else:
            self.upsample = None

        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=bias)

        if norm:
            self.norm2 = nn.BatchNorm2d(out_channel)

        self.activation2 = get_activation(activation)

        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=bias)

        if downsample:
            self.downsample = nn.AvgPool2d(2)

        else:
            self.downsample = None

        self.skip = None

        if in_channel != out_channel or upsample or downsample:
            self.skip = nn.Conv2d(in_channel, out_channel, 1, bias=False)

    def forward(self, input):
        out = input

        if self.norm:
            out = self.norm1(out)

        if not self.first:
            out = self.activation1(out)

        if self.upsample:
            out = self.upsample(out)

        out = self.conv1(out)

        if self.norm:
            out = self.norm2(out)

        out = self.activation2(out)
        out = self.conv2(out)

        if self.downsample:
            out = self.downsample(out)

        skip = input

        if self.skip is not None:
            if self.upsample:
                skip = self.upsample(skip)

            if self.downsample and self.first:
                skip = self.downsample(skip)

            skip = self.skip(skip)

            if self.downsample and not self.first:
                skip = self.downsample(skip)

        return out + skip


class SelfAttention(nn.Module):
    def __init__(self, in_channel, divider=8):
        super().__init__()

        self.query = nn.Conv2d(in_channel, in_channel // divider, 1, bias=False)
        self.key = nn.Conv2d(in_channel, in_channel // divider, 1, bias=False)
        self.value = nn.Conv2d(in_channel, in_channel // 2, 1, bias=False)
        self.out = nn.Conv2d(in_channel // 2, in_channel, 1, bias=False)
        self.divider = divider
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        batch, channel, height, width = input.shape

        query = (
            self.query(input)
            .view(batch, channel // self.divider, height * width)
            .transpose(1, 2)
        )
        key = F.max_pool2d(self.key(input), 2).view(
            batch, channel // self.divider, height * width // 4
        )
        value = F.max_pool2d(self.value(input), 2).view(
            batch, channel // 2, height * width // 4
        )
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 2)
        attn = torch.bmm(value, attn.transpose(1, 2)).view(
            batch, channel // 2, height, width
        )
        attn = self.out(attn)
        out = self.gamma * attn + input

        return out


class Generator(nn.Module):
    def __init__(
        self,
        feature_channels=(64, 128, 256, 512, 512),
        channel_multiplier=64,
        channels=(8, 8, 4, 2, 2, 1),
        blocks='rrrrr',
        upsample='nuuuu',
        activation='relu',
        feature_kernel_size=1,
    ):
        super().__init__()

        self.n_resblock = len([c for c in blocks if c == 'r'])
        self.use_affine = [b == 'r' for b in blocks]

        self.encoder = UNet()

        feat_i = 4

        self.blocks = nn.ModuleList()
        self.feature_blocks = nn.ModuleList()

        in_channel = channels[0] * channel_multiplier
        for block, ch, up in zip(blocks, channels, upsample):
            if block == 'r':
                self.blocks.append(
                    ResBlock(
                        in_channel,
                        ch * channel_multiplier,
                        norm=False,
                        upsample=up == 'u',
                        activation=activation,
                    )
                )

                self.feature_blocks.append(
                    nn.Conv2d(
                        feature_channels[feat_i],
                        ch * channel_multiplier,
                        feature_kernel_size,
                        padding=(feature_kernel_size - 1) // 2,
                        bias=False,
                    )
                )

                feat_i -= 1

            elif block == 'a':
                self.blocks.append(SelfAttention(in_channel))

            in_channel = ch * channel_multiplier

        self.colorize = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            get_activation(activation),
            nn.Conv2d(in_channel, 3, 3, padding=1),
            # nn.Tanh(),
        )

    def forward(self, input, features, mask=None):

        # batch_size = input.shape[0]
        #

        fea = self.encoder(input)

        feat_i = len(features) - 1
        layer_i = feat_i
        out = fea

        for affine, block in zip(self.use_affine, self.blocks):
            # print(out.shape)
            if affine:
                out = block(out)
                # print(out.shape, features[feat_i].shape, masks[feat_i].shape)
                out = out + self.feature_blocks[layer_i - feat_i](features[feat_i])
                feat_i -= 1

            else:
                out = block(out)

        out = self.colorize(out)
        return out


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)