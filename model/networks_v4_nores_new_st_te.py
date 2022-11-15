import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import numpy as np
from torch.optim import lr_scheduler
import util.util as util
from util.Selfpatch import Selfpatch
###############################################################################
# Functions
###############################################################################


def get_nonlinearity_layer(activation_type='PReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU(True)
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU(True)
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.2, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_net(net, init_type='normal', gpu_ids=[]):

    # print_network(net)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net = torch.nn.DataParallel(net, gpu_ids)
        net.cuda()
    init_weights(net, init_type)
    return net


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch+1+1+opt.epoch_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.uniform_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def _freeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def freeze_in(m):
    classname = m.__class__.__name__
    if classname.find('InstanceNorm') != -1:
        m.eval()
        #m.weight.requires_grad = False
        #m.bias.requires_grad = False


######################################################################################
# Basic Operation
######################################################################################

class GaussianNoiseLayer(nn.Module):
    def __init__(self):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable((torch.randn(x.size()).cuda(x.data.get_device()) - 0.5) / 10.0)
        return x+noise


class _InceptionBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), width=1, drop_rate=0, use_bias=False):
        super(_InceptionBlock, self).__init__()

        self.width = width
        self.drop_rate = drop_rate

        for i in range(width):
            layer = nn.Sequential(
                nn.ReflectionPad2d(i*2+1),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=0, dilation=i*2+1, bias=use_bias)
            )
            setattr(self, 'layer'+str(i), layer)

        self.norm1 = norm_layer(output_nc * width)
        self.norm2 = norm_layer(output_nc)
        self.nonlinearity = nonlinearity
        self.branch1x1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(output_nc * width, output_nc, kernel_size=3, padding=0, bias=use_bias)
        )

    def forward(self, x):
        result = []
        for i in range(self.width):
            layer = getattr(self, 'layer'+str(i))
            result.append(layer(x))
        output = torch.cat(result, 1)
        output = self.nonlinearity(self.norm1(output))
        output = self.norm2(self.branch1x1(output))
        if self.drop_rate > 0:
            output = F.dropout(output, p=self.drop_rate, training=self.training)

        return self.nonlinearity(output+x)


class _EncoderBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_EncoderBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.Conv2d(middle_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



class _DecoderUpBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_DecoderUpBlock, self).__init__()

        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.ConvTranspose2d(middle_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _OutputBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, use_bias=False):
        super(_OutputBlock, self).__init__()

        model = [
            nn.ReflectionPad2d(int(kernel_size/2)),
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=0, bias=use_bias),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _BasicBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_BasicBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity,
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FeaBlock(nn.Module):
    def __init__(self, C_in, C_out):
        super(FeaBlock, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_out, 1, 1, 0)
        self.bn = nn.BatchNorm2d(C_out)
        self.act = nn.PReLU()
        self.attention = SE_Block(C_out, reduction=8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.attention(out)
        return out


class _UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, layers=4, norm='batch', activation='PReLU', drop_rate=0, add_noise=False, gpu_ids=[],
                 weight=0.1):
        super(_UNetGenerator, self).__init__()

        self.gpu_ids = gpu_ids
        self.layers = layers
        self.weight = weight
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # encoder part
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nonlinearity
        )
        self.conv2 = _EncoderBlock(ngf, ngf*2, ngf*2, norm_layer, nonlinearity, use_bias)
        self.conv3 = _EncoderBlock(ngf*2, ngf*4, ngf*4, norm_layer, nonlinearity, use_bias)
        self.conv4 = _EncoderBlock(ngf*4, ngf*8, ngf*8, norm_layer, nonlinearity, use_bias)

        for i in range(layers-4):
            conv = _EncoderBlock(ngf*8, ngf*8, ngf*8, norm_layer, nonlinearity, use_bias)
            setattr(self, 'down'+str(i), conv.model)

        center=[]
        for i in range(7-layers):
            center +=[
                _InceptionBlock(ngf*8, ngf*8, norm_layer, nonlinearity, 7-layers, drop_rate, use_bias)
            ]

        center += [
        _DecoderUpBlock(ngf*8, ngf*8, ngf*4, norm_layer, nonlinearity, use_bias)
        ]
        if add_noise:
            center += [GaussianNoiseLayer()]
        self.center = nn.Sequential(*center)

        for i in range(layers-4):
            upconv = _DecoderUpBlock(ngf*(8+4), ngf*8, ngf*4, norm_layer, nonlinearity, use_bias)
            setattr(self, 'up' + str(i), upconv.model)

        self.deconv4 = _DecoderUpBlock(ngf*(4+4), ngf*8, ngf*2, norm_layer, nonlinearity, use_bias)
        self.deconv3 = _DecoderUpBlock(ngf*(2+2)+output_nc, ngf*4, ngf, norm_layer, nonlinearity, use_bias)
        self.deconv2 = _DecoderUpBlock(ngf*(1+1)+output_nc, ngf*2, int(ngf/2), norm_layer, nonlinearity, use_bias)

        self.output4 = _OutputBlock(ngf*(4+4), output_nc, 3, use_bias)
        self.output3 = _OutputBlock(ngf*(2+2)+output_nc, output_nc, 3, use_bias)
        self.output2 = _OutputBlock(ngf*(1+1)+output_nc, output_nc, 3, use_bias)
        self.output1 = _OutputBlock(int(ngf/2)+output_nc, output_nc, 7, use_bias)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.fea1 = FeaBlock(ngf, ngf)
        self.fea2 = FeaBlock(ngf*2, ngf*2)
        self.fea3 = FeaBlock(ngf*4, ngf*4)
        self.fea4 = FeaBlock(ngf * 8, ngf * 8)


    def forward(self, input, fea_vgg):
        result = []
        add_fea_1 = self.fea1(fea_vgg[0])
        result.append(add_fea_1)
        conv1 = self.pool(self.conv1(input)+add_fea_1)
        add_fea_2 = self.fea2(fea_vgg[1])
        result.append(add_fea_2)
        conv2 = self.pool(self.conv2.forward(conv1)+add_fea_2)
        add_fea_3 = self.fea3(fea_vgg[2])
        result.append(add_fea_3)
        conv3 = self.pool(self.conv3.forward(conv2)+add_fea_3)
        add_fea_4 = self.fea4(fea_vgg[3])
        result.append(add_fea_4)
        center_in = self.pool(self.conv4.forward(conv3)+add_fea_4)

        middle = [center_in]
        for i in range(self.layers-4):
            model = getattr(self, 'down'+str(i))
            center_in = self.pool(model.forward(center_in))
            middle.append(center_in)
        center_out = self.center.forward(center_in)


        for i in range(self.layers-4):
            model = getattr(self, 'up'+str(i))
            center_out = model.forward(torch.cat([center_out, middle[self.layers-5-i]], 1))

        ans = torch.cat([center_out, conv3 * self.weight],1)
        deconv4 = self.deconv4.forward(torch.cat([center_out, conv3 * self.weight], 1))
        output4 = self.output4.forward(torch.cat([center_out, conv3 * self.weight], 1))

        deconv3 = self.deconv3.forward(torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1))
        output3 = self.output3.forward(torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1))

        deconv2 = self.deconv2.forward(torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1))
        output2 = self.output2.forward(torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1))

        output1 = self.output1.forward(torch.cat([deconv2, self.upsample(output2)], 1))


        return output1, result

class Downblock(nn.Module):
    def __init__(self, C_in, C_out):
        super(Downblock, self).__init__()
        self.down = nn.Conv2d(C_in, C_out, 3, 2, 1)

    def forward(self, x):
        return self.down(x)


class Upblock(nn.Module):
    def __init__(self, C_in, C_out,norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU()):
        super(Upblock, self).__init__()

        model = [
            nn.ConvTranspose2d(C_in, C_out, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(C_out),
            nonlinearity
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class ConvDown(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0, dilation=1, groups=1, bias=False, layers=1, activ=True):
        super().__init__()
        nf_mult = 1
        nums = out_c / 64
        sequence = []

        for i in range(1, layers + 1):
            nf_mult_prev = nf_mult
            if nums == 8:
                if in_c == 512:

                    nfmult = 1
                else:
                    nf_mult = 2

            else:
                nf_mult = min(2 ** i, 8)
            if kernel != 1:

                if activ == False and layers == 1:
                    sequence += [
                        nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                  kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                        nn.InstanceNorm2d(nf_mult * in_c)
                    ]
                else:
                    sequence += [
                        nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                  kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                        nn.InstanceNorm2d(nf_mult * in_c),
                        nn.LeakyReLU(0.2, True)
                    ]

            else:

                sequence += [
                    nn.Conv2d(in_c, out_c,
                              kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                    nn.InstanceNorm2d(out_c),
                    nn.LeakyReLU(0.2, True)
                ]

            if activ == False:
                if i + 1 == layers:
                    if layers == 2:
                        sequence += [
                            nn.Conv2d(nf_mult * in_c, nf_mult * in_c,
                                      kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                            nn.InstanceNorm2d(nf_mult * in_c)
                        ]
                    else:
                        sequence += [
                            nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                      kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                            nn.InstanceNorm2d(nf_mult * in_c)
                        ]
                    break

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class ConvUp(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel,
                              stride, padding, dilation, groups, bias)
        self.bn = nn.InstanceNorm2d(out_c)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, size):
        out = F.interpolate(input=input, size=size, mode='bilinear')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True)) # 32 -> 64
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True)) # 32 -> 64
        return x[0] * (scale + 1) + shift

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        # self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
        #                            stride, padding, dilation, groups, False)

        # torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        # for param in self.mask_conv.parameters():
        #     param.requires_grad = False

    def forward(self, inputt):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        # input = inputt[0]
        # mask = inputt[1].float().cuda()
        input = inputt

        # output = self.input_conv(input * mask)
        output = self.input_conv(input)
        out = output
        # if self.input_conv.bias is not None:
        #     output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
        #         output)
        # else:
        #     output_bias = torch.zeros_like(output)
        #
        # with torch.no_grad():
        #     output_mask = self.mask_conv(mask)
        #
        # no_update_holes = output_mask == 0
        # mask_sum = output_mask.masked_fill_(no_update_holes.bool(), 1.0)
        # output_pre = (output - output_bias) / mask_sum + output_bias
        # output = output_pre.masked_fill_(no_update_holes.bool(), 0.0)
        # new_mask = torch.ones_like(output)
        # new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
        # out = []
        # out.append(output)
        # out.append(new_mask)
        return out

class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='leaky',
                 conv_bias=False, innorm=False, inner=False, outer=False):
        super().__init__()
        if sample == 'same-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == 'same-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.innorm = innorm
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out = self.bn(out)
            out = self.activation(out)
            out = self.conv(out)
            out = self.bn(out)
            out = self.activation(out)

        elif self.innorm:
            out = self.conv(out)
            out = self.bn(out)
            out = self.activation(out)
        elif self.outer:
            out = self.conv(out)
            out = self.bn(out)
        else:
            out = self.conv(out)
            out = self.bn(out)
            if hasattr(self, 'activation'):
                out = self.activation(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return x * y.expand_as(x)

class BASE(nn.Module):
    def __init__(self, inner_nc):
        super(BASE, self).__init__()
        se = SELayer(inner_nc, 16)
        model = [se]
        gus = util.gussin(1.5).cuda()
        self.gus = torch.unsqueeze(gus, 1).double()
        self.model = nn.Sequential(*model)
        self.down = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        Nonparm = Selfpatch()
        out_32 = self.model(x)
        b, c, h, w = out_32.size()
        gus = self.gus.float()
        gus_out = out_32[0].expand(h * w, c, h, w)
        gus_out = gus * gus_out
        gus_out = torch.sum(gus_out, -1)
        gus_out = torch.sum(gus_out, -1)
        gus_out = gus_out.contiguous().view(b, c, h, w)
        csa2_in = F.sigmoid(out_32)
        csa2_f = torch.nn.functional.pad(csa2_in, (1, 1, 1, 1))
        csa2_ff = torch.nn.functional.pad(out_32, (1, 1, 1, 1))
        csa2_fff, csa2_f, csa2_conv = Nonparm.buildAutoencoder(csa2_f[0], csa2_in[0], csa2_ff[0], 3, 1)
        csa2_conv = csa2_conv.expand_as(csa2_f)
        csa_a = csa2_conv * csa2_f
        csa_a = torch.mean(csa_a, 1)
        a_c, a_h, a_w = csa_a.size()
        csa_a = csa_a.contiguous().view(a_c, -1)
        csa_a = F.softmax(csa_a, dim=1)
        csa_a = csa_a.contiguous().view(a_c, 1, a_h, a_h)
        out = csa_a * csa2_fff
        out = torch.sum(out, -1)
        out = torch.sum(out, -1)
        out_csa = out.contiguous().view(b, c, h, w)
        out_32 = torch.cat([gus_out, out_csa], 1)
        out_32 = self.down(out_32)
        return out_32

class UnderNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, center_num=3, norm='batch', activation='PReLU', drop_rate=0, gpu_ids=[], weight=1.0):
        super(UnderNet, self).__init__()

        self.weight = weight
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            nonlinearity
        )
        self.conv1 = _BasicBlock(ngf*2, ngf, norm_layer, nonlinearity, use_bias)
        # *2
        self.down1 = Downblock(ngf, ngf*2)
        self.conv2 = _BasicBlock(ngf*4, ngf * 2, norm_layer, nonlinearity, use_bias)
        # *4
        self.down2 = Downblock(ngf*2, ngf*4)
        self.conv3 = _BasicBlock(ngf*8, ngf * 4, norm_layer, nonlinearity, use_bias)
        # *8
        self.down3 = Downblock(ngf*4, ngf*8)
        self.conv4 = _BasicBlock(ngf*16, ngf * 8, norm_layer, nonlinearity, use_bias)
        # *16
        self.down4 = Downblock(ngf*8, ngf*8)

        center=[]
        for i in range(center_num):
            center += [
                _InceptionBlock(ngf * 8, ngf * 8, norm_layer, nonlinearity, center_num, drop_rate, use_bias)
            ]
        self.center = nn.Sequential(*center)

        self.up4 = Upblock(ngf*8, ngf*8)
        self.de4 = _BasicBlock(ngf*16, ngf * 8, norm_layer, nonlinearity, use_bias)
        self.up3 = Upblock(ngf*8, ngf*4)
        self.de3 = _BasicBlock(ngf * 8, ngf * 4, norm_layer, nonlinearity, use_bias)
        self.up2 = Upblock(ngf*4, ngf*2)
        self.de2 = _BasicBlock(ngf*4, ngf*2, norm_layer, nonlinearity, use_bias)
        self.up1 = Upblock(ngf*2, ngf)
        self.de1 = _BasicBlock(ngf*2, ngf, norm_layer, nonlinearity, use_bias)

        self.out = _OutputBlock(ngf, output_nc, 7, use_bias)

        self.fea1 = FeaBlock(64, ngf)
        self.fea2 = FeaBlock(128, ngf*2)
        self.fea3 = FeaBlock(256, ngf*4)
        self.fea4 = FeaBlock(512, ngf * 8)

        self.fea1_1 = Upblock(128, 64)
        self.fea3_1 = ConvDown(128, 256, 4, 2, padding=1)
        self.fea4_1 = ConvDown(128, 256, 4, 2, padding=1)
        self.fea4_2 = ConvDown(256, 512, 4, 2, padding=1)

        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.down_1 = ConvDown(64, 128, 4, 2, padding=1)
        self.down_2 = ConvDown(128, 128, 1, 1)

        self.up_2 = Upblock(256, 128)
        self.up_3 = Upblock(512, 256)
        self.up_4 = Upblock(256, 128)

        self.down_DE = ConvDown(256, 128, 1, 1)
        self.down_ST = ConvDown(256, 128, 1, 1)
        self.down_fus_DE = ConvDown(384, 128, 1, 1)
        self.down_fus_ST = ConvDown(384, 128, 1, 1)
        self.fuse = ConvDown(256, 128, 1, 1)

        seuqence_3 = []
        seuqence_5 = []
        seuqence_7 = []
        for i in range(3):
            seuqence_3 += [PCBActiv(128, 128, innorm=True)]
            seuqence_5 += [PCBActiv(128, 128, sample='same-5', innorm=True)]
            seuqence_7 += [PCBActiv(128, 128, sample='same-7', innorm=True)]

        self.cov_3 = nn.Sequential(*seuqence_3)
        self.cov_5 = nn.Sequential(*seuqence_5)
        self.cov_7 = nn.Sequential(*seuqence_7)

        self.base = BASE(256)

    def forward(self, x, fea_vgg):
        # TODO: Features from VGG module
        x1 = self.activation(fea_vgg[0])
        x2 = self.activation(fea_vgg[1])
        x3 = self.activation(fea_vgg[2])
        x4 = self.activation(fea_vgg[3])

        x1_1 = self.down_1(x1) # torch.Size([8, 128, 112, 112])
        x2_1 = self.down_2(x2)
        x3_1 = self.up_2(x3)
        x4_1 = self.up_4(self.up_3(x4))


        # TODO: Feature Selector
        x_DE = torch.cat([x1_1, x2_1], 1) # torch.Size([8, 256, 112, 112])
        x_ST = torch.cat([x3_1, x4_1], 1) # torch.Size([8, 256, 112, 112])

        x_DE = self.down_DE(x_DE) #torch.Size([8, 128, 112, 112])
        x_ST = self.down_ST(x_ST) #torch.Size([8, 128, 112, 112])


        # TODO: Multi-path CFRM for Details
        x_DE_3 = self.cov_3(x_DE)  # torch.Size([8, 128, 112, 112])
        x_DE_5 = self.cov_5(x_DE)  # torch.Size([8, 128, 112, 112])
        x_DE_7 = self.cov_7(x_DE)  # torch.Size([8, 128, 112, 112])
        x_DE_fuse = torch.cat([x_DE_3, x_DE_5, x_DE_7], 1) # torch.Size([8, 384, 112, 112])
        x_DE_fi = self.down_fus_DE(x_DE_fuse)  # torch.Size([8, 128, 112, 112])

        # TODO: Multi-path CFRM for Structures
        x_ST_3 = self.cov_3(x_ST)  # torch.Size([8, 128, 112, 112])
        x_ST_5 = self.cov_5(x_ST)  # torch.Size([8, 128, 112, 112])
        x_ST_7 = self.cov_7(x_ST)  # torch.Size([8, 128, 112, 112])
        x_ST_fuse = torch.cat([x_ST_3, x_ST_5, x_ST_7], 1)  # torch.Size([8, 384, 112, 112])
        x_ST_fi = self.down_fus_DE(x_ST_fuse)  # torch.Size([8, 128, 112, 112])

        # TODO: Fusion of Details and Structures
        x_cat = torch.cat([x_ST_fi, x_DE_fi], 1)
        x_cat_fuse = self.fuse(x_cat)  # torch.Size([8, 128, 112, 112])

        # TODO: Encoder
        # level 1
        add_fea_1 = self.fea1(self.fea1_1(x_cat_fuse))  # torch.Size([8, 32, 224, 224])
        conv1 = self.conv1(torch.cat([self.head(x), add_fea_1], dim=1))

        # level 2
        add_fea_2 = self.fea2(x_cat_fuse) # torch.Size([8, 64, 112, 112])
        conv2_1 = self.down1(conv1)
        conv2 = self.conv2(torch.cat([conv2_1, add_fea_2], dim=1)) # torch.Size([8, 64, 112, 112])

        # level 3
        add_fea_3 = self.fea3(self.fea3_1(x_cat_fuse)) # torch.Size([8, 128, 56, 56])
        conv3_2 = self.down2(conv2)
        conv3 = self.conv3(torch.cat([conv3_2, add_fea_3], dim=1)) # torch.Size([8, 128, 56, 56])

        # level 4

        add_fea_4 = self.fea4(self.fea4_2(self.fea4_1(x_cat_fuse))) # torch.Size([8, 256, 28, 28])
        conv4_3 = self.down3(conv3)
        conv4 = self.conv4(torch.cat([conv4_3, add_fea_4], dim=1))


        # TODO: Bottom
        center = self.down4(conv4)
        center = self.center(center)

        # TODO: Decoder
        # level 4
        deco4 = self.de4(torch.cat([self.up4(center), self.weight * conv4], dim=1))

        # level 3
        deco3 = self.de3(torch.cat([self.up3(deco4), 0.8 * self.weight * conv3], dim=1))

        # level 2
        deco2 = self.de2(torch.cat([self.up2(deco3), 0.4 * self.weight * conv2], dim=1))

        # level 1
        deco1 = self.de1(torch.cat([self.up1(deco2), 0.2 * self.weight * conv1], dim=1))

        out = self.out(deco1)

        return out