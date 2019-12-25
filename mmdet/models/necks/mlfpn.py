import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init as init

from ..registry import NECKS


def conv_ws_2d(input,
               weight,
               bias=None,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               eps=1e-5):
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class ConvWS2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5):
        super(ConvWS2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps

    def forward(self, x):
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups, self.eps)


class BasicConv(nn.Module):
    """
    Kai: In Article, one block is composed of "Conv + BN + ReLU"
    """

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = ConvWS2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.bn = nn.GroupNorm(out_planes // 16, out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class TUM(nn.Module):

    def __init__(self,
                 first_level=True,
                 input_planes=128,
                 is_smooth=True,
                 side_channel=512,
                 scales=6,
                 ssd_style_tum=True):   # This param is added by coder
        super(TUM, self).__init__()
        self.is_smooth = is_smooth
        self.side_channel = side_channel
        self.input_planes = input_planes
        self.planes = 2 * self.input_planes     # important
        self.first_level = first_level
        self.scales = scales
        if first_level:
            self.in1 = input_planes             # 128
        else:
            self.in1 = input_planes + side_channel      # 128 + 512
        self.layers = nn.Sequential()
        self.layers.add_module('{}'.format(len(self.layers)),
                               BasicConv(self.in1, self.planes, 3, 2, 1))
        if ssd_style_tum:
            for i in range(self.scales - 2):
                if not i == self.scales - 3:
                    self.layers.add_module(
                        '{}'.format(len(self.layers)),
                        BasicConv(self.planes, self.planes, 3, 2, 1))
                else:
                    self.layers.add_module(
                        '{}'.format(len(self.layers)),
                        BasicConv(self.planes, self.planes, 3, 1, 0))
        else:
            for i in range(self.scales - 2):
                self.layers.add_module(
                    '{}'.format(len(self.layers)),
                    BasicConv(self.planes, self.planes, 3, 2, 1))
        self.toplayer = nn.Sequential(
            BasicConv(self.planes, self.planes, 1, 1, 0))

        self.latlayer = nn.Sequential()
        for i in range(self.scales - 2):
            self.latlayer.add_module(
                '{}'.format(len(self.latlayer)),
                BasicConv(self.planes, self.planes, 3, 1, 1))
        self.latlayer.add_module('{}'.format(len(self.latlayer)),
                                 BasicConv(self.in1, self.planes, 3, 1, 1))

        if self.is_smooth:
            smooth = list()
            for i in range(self.scales - 1):
                smooth.append(BasicConv(self.planes, self.planes, 1, 1, 0))
            self.smooth = nn.Sequential(*smooth)

    def _upsample_add(self, x, y, fuse_type='interp'):
        _, _, H, W = y.size()
        if fuse_type == 'interp':
            return F.interpolate(x, size=(H, W), mode='nearest') + y
        else:
            raise NotImplementedError
            # return nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)

    def forward(self, x, y):
        if not self.first_level:
            x = torch.cat([x, y], 1)
        conved_feat = [x]
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            conved_feat.append(x)

        deconved_feat = [self.toplayer[0](conved_feat[-1])]
        for i in range(len(self.latlayer)):
            deconved_feat.append(
                self._upsample_add(
                    deconved_feat[i],
                    self.latlayer[i](conved_feat[len(self.layers) - 1 - i])))
        if self.is_smooth:
            smoothed_feat = [deconved_feat[0]]
            for i in range(len(self.smooth)):
                smoothed_feat.append(self.smooth[i](deconved_feat[i + 1]))
            return smoothed_feat
        return deconved_feat


class SFAM(nn.Module):
    def __init__(self, planes, num_levels, num_scales, compress_ratio=16):
        super(SFAM, self).__init__()
        self.planes = planes
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.compress_ratio = compress_ratio

        self.fc1 = nn.ModuleList([nn.Conv2d(self.planes * self.num_levels,
                                            self.planes * self.num_levels // 16,
                                            1, 1, 0)] * self.num_scales)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList([nn.Conv2d(self.planes * self.num_levels // 16,
                                            self.planes * self.num_levels,
                                            1, 1, 0)] * self.num_scales)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        attention_feat = []
        for i, _mf in enumerate(x):
            _tmp_f = self.avgpool(_mf)
            _tmp_f = self.fc1[i](_tmp_f)
            _tmp_f = self.relu(_tmp_f)
            _tmp_f = self.fc2[i](_tmp_f)
            _tmp_f = self.sigmoid(_tmp_f)
            attention_feat.append(_mf*_tmp_f)
        return attention_feat


dim_conv = [256, 512, 1024, 2048]   # 对应Resnet50每个stage的out-chs？


@NECKS.register_module
class MLFPN(nn.Module):
    """
    Args:
        backbone_choice: 'ResNet' or 'VGG'. About VGG, it need to be designed by me.
        scale_outs_num: each TUM will output features with scales.
        scale_outs_num: the num of scale obtained by each TUM
        tum_num: the num of TUM module:
        smooth: the param is used in TUM
        base_feature_size: ?
        base_choice: [int], 表示使用backbone几个stage的feature
        base_list: [list], 表示使用backbone哪几个stage的feature,从1计数
        norm: [bool], used in SFAM
        ssd_style_tum: [bool], ?
        out_indices: [list], add by kai
        sfam: [bool], whether to use sfam module
    """

    def __init__(
            self,
            backbone_choice,
            in_channels,
            planes,                 # internal plane size, 256
            scale_outs_num,         # scale_feature_num, 6
            tum_num,                # tum_num, 8
            smooth,                 # smooth
            base_feature_size,      # ?
            base_choice,            # ?
            base_list,              # base_list=[2, 3]
            norm,
            ssd_style_tum=True,
            out_indices=None,       # kai add the param
            sfam=True               # kai add the param
    ):
        super(MLFPN, self).__init__()
        # print(type(base_list))
        # print("config base_list ******** {}".format(base_list))
        self.input_channel = in_channels
        self.num_levels = tum_num
        self.planes = planes
        self.smooth = smooth
        self.num_scales = scale_outs_num
        self.base_feature_size = base_feature_size
        self.base_choice = base_choice
        self.base_list = base_list
        self.norm = norm
        self.backbone_choice = backbone_choice
        self.ssd_style_tum = ssd_style_tum
        self.out_indices = out_indices
        self.sfam = sfam
        # print(self.base_list[1])

        if base_choice == 1:
            self.dim = dim_conv[self.base_list[0]]
        else:
            # construct base features
            if self.backbone_choice == 'ResNet':
                self.shallow_in = dim_conv[self.base_list[0] - 1]   # 512
                self.deep_in = dim_conv[self.base_list[1] - 1]      # 1024
                self.shallow_out = 256
                self.deep_out = 512
            else:
                self.shallow_in = in_channels[0]
                self.deep_in = in_channels[1]
                self.shallow_out = 256
                self.deep_out = 512

            self.reduce = BasicConv(
                self.shallow_in,    # 512
                self.shallow_out,   # 256
                kernel_size=3,
                stride=1,
                padding=1)
            self.up_reduce = BasicConv(     # 1024 -> 512
                self.deep_in, self.deep_out, kernel_size=1, stride=1)

            # construct others
            self.Norm = nn.BatchNorm2d(256 * 4)     # 8?
            self.leach = nn.ModuleList([
                BasicConv(
                    self.deep_out + self.shallow_out,   # 768
                    self.planes // 2,                   # 128
                    kernel_size=(1, 1),
                    stride=(1, 1))
            ] * self.num_levels)

        # construct tums
        for i in range(self.num_levels):
            if i == 0:
                setattr(
                    self, 'unet{}'.format(i + 1),
                    TUM(first_level=True,                   # 是否是第一级TUM
                        input_planes=self.planes // 2,      # 128
                        is_smooth=self.smooth,
                        scales=self.num_scales,
                        side_channel=512,
                        ssd_style_tum=self.ssd_style_tum))
            else:
                setattr(
                    self, 'unet{}'.format(i + 1),
                    TUM(first_level=False,
                        input_planes=self.planes // 2,
                        is_smooth=self.smooth,
                        scales=self.num_scales,
                        side_channel=self.planes,
                        ssd_style_tum=self.ssd_style_tum))

        # construct SFAM module
        if self.sfam:
            self.sfam_module = SFAM(self.planes, self.num_levels, self.num_scales, compress_ratio=16)

    def init_weights(self):
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(
                        self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def forward(self, input):
        if self.base_feature_size == 4:
            if self.base_choice == 1:
                base_feature = input[self.base_list[0]]
            elif self.base_choice == 2:
                if self.backbone_choice == 'ResNet':
                    """
                    Edited by Kai: when base_feature_size==4 & base_choice==2 & backbone_choice=='ResNet',
                    the following 'if' code will be performed. 
                    Actually, this part is FFMv1 to obtain base features.           
                    """
                    base_feature = torch.cat(
                        (self.reduce(input[self.base_list[0] - 1]),         # shallow_out=256
                         F.interpolate(
                             self.up_reduce(input[self.base_list[1] - 1]),
                             scale_factor=2,
                             mode='nearest')),                              # deep_out=512
                        1)
                else:
                    base_feature = torch.cat((self.reduce(input[0]),
                                              F.interpolate(
                                                  self.up_reduce(input[1]),
                                                  scale_factor=2,
                                                  mode='nearest')), 1)

            """
            Edited by Kai: this part is used to build TUM. self.num_levels = tum_num, 
            this parm denote the num of TUM we used.
            """
            # tum_outs: is the multi-level multi-scale feature
            tum_outs = [
                getattr(self, 'unet{}'.format(1))(self.leach[0](base_feature),
                                                  'none')
            ]
            for i in range(1, self.num_levels, 1):
                tum_outs.append(
                    getattr(self, 'unet{}'.format(i + 1))(
                        self.leach[i](base_feature), tum_outs[i - 1][-1]))

            """
            Edited by Kai: this part should be SFAM. But, it only has a scale-wise feature 
            cat operation, no an adaptive attention mechanism.
            """
            # concat with same scales
            sources = [
                torch.cat([_fx[i - 1] for _fx in tum_outs], 1)
                for i in range(self.num_scales, 0, -1)
            ]               # 使用双重for循环concat不同level, 相同scale的feature maps
            output = []     # the dim of output=num_scales=scale_outs_num

            # forward_sfam
            if self.sfam:
                sources = self.sfam_module(sources)
            sources[0] = self.Norm(sources[0])

            for i in range(0, self.num_scales, 1):
                output.append(sources[i])        # use 4,8,16,32,64
            # return tuple(output)

            """
            Kai add following lines for performing 'finest layer of MLFPN' experiment.
            """
            if self.out_indices is not None:
                return tuple([output[i] for i in self.out_indices])
            else:
                return tuple(output)
