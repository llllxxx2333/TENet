import torch
import torch.nn as nn
import torch.functional as F
from mobilenetv2 import mobilenet_v2
import numpy as np

def conv_bn(in_channels, out_channels, rate):
    if rate == 1:
        kernel_size = 3
        padding = 3
        dilation = 3
    else:
        kernel_size = 1
        padding = 0
        dilation = 1
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding,dilation=dilation,bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    return result

class Rep(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,stride = 1,
                  padding=3, dilation=3, groups=1, padding_mode='zeros', deploy=False):
        super(Rep, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.nonlinearity = nn.ReLU()


        self.GAP_Conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.Sigmoid()
        )

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=1,
                                         padding=padding, dilation=dilation, groups=groups,
                                         bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                      rate=1)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                   rate=0)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
             return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                         out_channels=self.rbr_dense.conv.out_channels,
                                         kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                         padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                         groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            out = self.nonlinearity(self.rbr_reparam(inputs))
        else:
            if self.rbr_identity is None:
                id_out = 0
            else:
                id_out = self.rbr_identity(inputs)
            out = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
            out = self.nonlinearity(out)


        y = out

        return y


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class CC(nn.Module):
    def __init__(self):
        super(CC, self).__init__()
        self.upsample1 = nn.Sequential(nn.Conv2d(80, 12, 3, 1, 1, ), nn.BatchNorm2d(12), nn.GELU())
        self.upsample2 = nn.Sequential(nn.Conv2d(128, 12, 3, 1, 1, ), nn.BatchNorm2d(12), nn.GELU(),
                                      nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample3 = nn.Sequential(nn.Conv2d(160, 12, 3, 1, 1, ), nn.BatchNorm2d(12), nn.GELU(),
                                       nn.UpsamplingBilinear2d(scale_factor=4, ))
        self.conv1 = nn.Conv2d(12,12,3,padding=3,dilation=3)
        self.conv2 = nn.Conv2d(12, 12, 3, padding=5, dilation=5)
        self.conv3 = nn.Conv2d(12, 12, 3, padding=7, dilation=7)

        self.GAP_Conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(12, 36, 1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.CBG = nn.Sequential(nn.Conv2d(36, 36, 3, 1, 1, ), nn.BatchNorm2d(36), nn.GELU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.cag3 = ChannelAttention(80)
        self.caf3 = ChannelAttention(80)
        self.cag4 = ChannelAttention(128)
        self.caf4 = ChannelAttention(128)
        self.cag5 = ChannelAttention(160)
        self.caf5 = ChannelAttention(160)
    def forward(self, f5,f4,f3,g5,g4,g3):
        fg3 = self.caf3(f3) * g3 + g3 + self.cag3(g3) * f3 + f3 #80 56
        fg4 = self.caf4(f4) * g4 + g4 + self.cag4(g4) * f4 + f4 #128 28
        fg5 = self.caf5(f5) * g5 + g5 + self.cag5(g5) * f5 + f5 #160 14
        fg3 = self.upsample1(fg3)#12 56
        fg4 = self.upsample2(fg4)#12 56
        fg5 = self.upsample3(fg5)#12 56
        s = fg3 * fg4 * fg5
        s = self.GAP_Conv(s)
        fg3 = self.conv1(fg3)
        fg4 = self.conv2(fg4)
        fg5 = self.conv3(fg5)
        z = torch.cat((fg3,fg4,fg5),dim=1)#36 56
        z = z * s
        z = self.CBG(z) #36 112
        return z

class SS(nn.Module):
    def __init__(self):
        super(SS, self).__init__()
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2, )
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2, )

        self.satt1 = SpatialAttention()
        self.satt2 = SpatialAttention()
    def forward(self, GF3,GF2,GF1):
        s1 = self.satt1(GF3)
        GF23 = torch.cat((GF3,GF2),dim=1)
        GF23 = s1 * GF23 + GF23
        GF3 = self.upsample1(GF3)
        s2 = self.satt2(GF3)
        GF13 = torch.cat((GF3,GF1),dim=1)
        GF13 = s2 * GF13 + GF13
        GF23 = self.upsample2(GF23)
        F = GF13 + GF23

        return F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3, padding=1, polish=True):
        super(SpatialAttention, self).__init__()

        kernel = torch.ones((kernel_size, kernel_size))  # 全为1的3*3卷积核 权重为1
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # 扩为4维
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        kernel2 = torch.ones((1, 1)) * (kernel_size * kernel_size)  # 1*1的卷积核 权重为9
        kernel2 = kernel2.unsqueeze(0).unsqueeze(0)  # 扩为4维
        self.weight2 = nn.Parameter(data=kernel2, requires_grad=False)

        self.polish = polish
        self.pad = padding
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
    def __call__(self, x):
        map1,_ = torch.max(x,dim=1, keepdim=True)
        map2 = torch.mean(x,dim=1,keepdim=True)
        map = map1 + map2
        x1 = torch.nn.functional.conv2d(map, self.weight, padding=self.pad)  # 3*3卷积
        x2 = torch.nn.functional.conv2d(map, self.weight2, padding=0)  # 1*1卷积

        xatt = torch.cat([x1,x2],dim=1)
        xatt,_ = torch.max(xatt,dim=1,keepdim=True)

        xatt = self.bn(xatt)
        xatt = self.relu(xatt)

        if self.polish:
            xatt[:, :, :, 0] = 0
            xatt[:, :, :, -1] = 0
            xatt[:, :, 0, :] = 0
            xatt[:, :, -1, :] = 0
        xatt = self.sigmoid(xatt)
        output = xatt
        return output



class Cross(nn.Module):
    def __init__(self):
        super(Cross, self).__init__()
        self.satt_x = SpatialAttention()

    def forward(self, x,y):
        F = self.satt_x(x) * y + y + x
        return F

class LNet(nn.Module):
    def __init__(self):
        super(LNet, self).__init__()
        # rgb,depth encode
        self.rgb_pretrained = mobilenet_v2()
        self.depth_pretrained = mobilenet_v2()
        self.thermal_pretrained = mobilenet_v2()

        # Upsample_model
        self.upsample1_f = nn.Sequential(nn.Conv2d(52, 36, 3, 1, 1, ), nn.BatchNorm2d(36), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample2_f = nn.Sequential(nn.Conv2d(104, 36, 3, 1, 1, ), nn.BatchNorm2d(36), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample3_f = nn.Sequential(nn.Conv2d(160, 80, 3, 1, 1, ), nn.BatchNorm2d(80), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample4_f = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1, ), nn.BatchNorm2d(128), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample5_f = nn.Sequential(nn.Conv2d(320, 160, 3, 1, 1, ), nn.BatchNorm2d(160), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample1_g = nn.Sequential(nn.Conv2d(52, 36, 3, 1, 1, ), nn.BatchNorm2d(36), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample2_g = nn.Sequential(nn.Conv2d(104, 36, 3, 1, 1, ), nn.BatchNorm2d(36), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample3_g = nn.Sequential(nn.Conv2d(160, 80, 3, 1, 1, ), nn.BatchNorm2d(80), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample4_g = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1, ), nn.BatchNorm2d(128), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample5_g = nn.Sequential(nn.Conv2d(320, 160, 3, 1, 1, ), nn.BatchNorm2d(160), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.Repf5 = Rep(160, 160)
        self.Repf4 = Rep(128, 128)
        self.Repf3 = Rep(80, 80)
        self.Repf2 = Rep(36, 36)
        self.Repf1 = Rep(36, 36)

        self.Repg5 = Rep(160, 160)
        self.Repg4 = Rep(128, 128)
        self.Repg3 = Rep(80, 80)
        self.Repg2 = Rep(36, 36)
        self.Repg1 = Rep(36, 36)

        self.crossVD1 = Cross()
        self.crossVD2 = Cross()
        self.crossVD3 = Cross()
        self.crossVD4 = Cross()
        self.crossVD5 = Cross()

        self.crossVT1 = Cross()
        self.crossVT2 = Cross()
        self.crossVT3 = Cross()
        self.crossVT4 = Cross()
        self.crossVT5 = Cross()

        self.cc = CC()
        self.ss = SS()

        self.conv_g = nn.Conv2d(36, 1, 1)
        self.conv_f = nn.Conv2d(36, 1, 1)
        self.conv_fg = nn.Conv2d(72, 1, 1)

    def forward(self, rgb,depth, ti):
        # rgb
        A1, A2, A3, A4, A5 = self.rgb_pretrained(rgb)
        # ti
        A1_t, A2_t, A3_t, A4_t, A5_t = self.thermal_pretrained(ti)
        # dep
        A1_d, A2_d, A3_d, A4_d, A5_d = self.depth_pretrained(depth)

        F5 = self.crossVT5(A5, A5_t)
        F4 = self.crossVT4(A4, A4_t)
        F3 = self.crossVT3(A3, A3_t)
        F2 = self.crossVT2(A2, A2_t)
        F1 = self.crossVT1(A1, A1_t)

        G5 = self.crossVD5(A5, A5_d)
        G4 = self.crossVD4(A4, A4_d)
        G3 = self.crossVD3(A3, A3_d)
        G2 = self.crossVD2(A2, A2_d)
        G1 = self.crossVD1(A1, A1_d)


        F5 = self.upsample5_f(F5) #160 14
        F5 = self.Repf5(F5)
        F4 = torch.cat((F4, F5), dim=1)

        F4 = self.upsample4_f(F4) #128 28
        F4 = self.Repf4(F4)
        F3 = torch.cat((F3, F4), dim=1)#


        F3 = self.upsample3_f(F3) #80 56
        F3 = self.Repf3(F3)


        F2 = torch.cat((F2, F3), dim=1)

        F2 = self.upsample2_f(F2) #36 112
        F1 = torch.cat((F1, F2), dim=1)#52

        F1 = self.upsample1_f(F1) #36 224

        G5 = self.upsample5_g(G5)  # 160 14
        G5 = self.Repg5(G5)
        G4 = torch.cat((G4, G5), dim=1)


        G4 = self.upsample4_g(G4)  # 128 28
        G4 = self.Repg4(G4)
        G3 = torch.cat((G3, G4), dim=1)


        G3 = self.upsample3_g(G3)  # 80 54
        G3 = self.Repg3(G3)
        G2 = torch.cat((G2, G3), dim=1)#104 112

        G2 = self.upsample2_g(G2)  # 36 112
        #G2 = self.Repg2(G2)
        G1 = torch.cat((G1, G2), dim=1)

        G1 = self.upsample1_g(G1)  # 36 224
        #G1 = self.Repg1(G1)

        GF2 = F2 + G2#36 112
        GF1 = F1 + G1#36 224
        GF3 = self.cc(F5,F4,F3,G5,G4,G3)#36 112
        out = self.ss(GF3,GF2,GF1)#72 224

        outf = self.conv_f(F1)
        outg = self.conv_g(G1)
        out  = self.conv_fg(out)

        if self.training:
            return out, outf,outg
        return out

