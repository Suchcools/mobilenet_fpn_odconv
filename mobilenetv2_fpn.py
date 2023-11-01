import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(in_channels=inp, out_channels=inp * expand_ratio,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # depthwise convolution via groups
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio,
                      kernel_size=3, stride=stride, padding=1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pointwise linear convolution
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=oup,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class MobileNetV2_dynamicFPN(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2_dynamicFPN, self).__init__()

        self.input_channel = int(32 * width_mult)
        self.width_mult = width_mult
        self.pool_size = (7, 7)
        last_channel = 1280

        # First layer
        self.first_layer = nn.Sequential(
            ConvBNReLU(3, self.input_channel, stride=2)
        )

        # Inverted residual blocks (each n layers)
        self.inverted_residual_setting = [
            {'expansion_factor': 1, 'width_factor': 16, 'n': 1, 'stride': 1},
            {'expansion_factor': 6, 'width_factor': 24, 'n': 2, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 32, 'n': 3, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 64, 'n': 4, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 96, 'n': 3, 'stride': 1},
            {'expansion_factor': 6, 'width_factor': 160, 'n': 3, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 320, 'n': 1, 'stride': 1},
        ]
        self.inverted_residual_blocks = nn.ModuleList(
            [self._make_inverted_residual_block(**setting)
             for setting in self.inverted_residual_setting])

        last_layer = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        self.last_layer = ConvBNReLU(self.input_channel, last_layer, kernel_size=1)
    
        self.output = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features= last_layer, out_features= num_classes, bias=True)
        )

        # reduce feature maps to one pixel
        # allows to upsample semantic information of every part of the image
        self.average_pool = nn.AdaptiveAvgPool2d(1)

        # Top layer
        # input channels = last width factor
        self.top_layer = nn.Conv2d(
            int(self.inverted_residual_setting[-1]['width_factor'] * self.width_mult),
            256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        # exclude last setting as this lateral connection is the the top layer
        # build layer only if resulution has decreases (stride > 1)
        self.lateral_setting = [setting for setting in self.inverted_residual_setting[:-1]
                                if setting['stride'] > 1]
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(int(setting['width_factor'] * self.width_mult),
                      256, kernel_size=1, stride=1, padding=0)
            for setting in self.lateral_setting])

        # Smooth layers
        # n = lateral layers + 1 for top layer
        self.smooth_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)] *
                                           (len(self.lateral_layers) + 1))
        
        
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 将特征图展平为一维向量
            nn.Linear(last_channel * 7 * 7, num_classes)  # 1280*7*7 是展平后的特征向量的大小，172 是分类的类别数
        )

        self._initialize_weights()

    def _make_inverted_residual_block(self, expansion_factor, width_factor, n, stride):
        inverted_residual_block = []
        output_channel = int(width_factor * self.width_mult)
        for i in range(n):
            # except the first layer, all layers have stride 1
            if i != 0:
                stride = 1
            inverted_residual_block.append(
                InvertedResidual(self.input_channel, output_channel, stride, expansion_factor))
            self.input_channel = output_channel

        return nn.Sequential(*inverted_residual_block)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        # bottom up
        x = self.first_layer(x)

        # loop through inverted_residual_blocks (mobile_netV2)
        # save lateral_connections to lateral_tensors
        # track how many lateral connections have been made
        lateral_tensors = []
        n_lateral_connections = 0
        for i, block in enumerate(self.inverted_residual_blocks):
            output = block(x)  # run block of mobile_net_V2
            if self.inverted_residual_setting[i]['stride'] > 1 \
                    and n_lateral_connections < len(self.lateral_layers):
                lateral_tensors.append(self.lateral_layers[n_lateral_connections](output))
                n_lateral_connections += 1
            x = output

        # 1 output
        # output = self.last_layer(output)
        # output = output.mean([2, 3])
        # output = self.output(output)

        # 2 Feature map

        # 将特征图的分辨率降到一个像素，从而得到一个特征图（通常称为P6），用于最高分辨率信息。

        x = self.average_pool(x) 


        # 构建特征金字塔网络。它包括一个顶层卷积层（top_layer），一系列横向连接（lateral_layers
        # 横向连接通过将底层特征图与高层特征图进行上采样和融合来构建特征金字塔的不同层。

        # connect m_layer with previous m_layer and lateral layers recursively
        m_layers = [self.top_layer(x)]
        # reverse lateral tensor order for top down
        lateral_tensors.reverse()
        for lateral_tensor in lateral_tensors:
            m_layers.append(self._upsample_add(m_layers[-1], lateral_tensor))


        # 平滑层用于平滑特征图以减小噪音。
        # smooth all m_layers
        assert len(self.smooth_layers) == len(m_layers)
        p_layers = [smooth_layer(m_layer) for smooth_layer, m_layer in zip(self.smooth_layers, m_layers)]
        
        pooled_maps = []
        # 是否特征融合输出
        for feat in p_layers:
            if tuple(feat.shape[-2:]) in [(14,14),(28,28)]:
                continue
            pooled_feat = F.adaptive_max_pool2d(feat, (7,7))
            pooled_maps.append(pooled_feat)
        # 将这些特征图连接在一起
        concatenated_features = torch.cat(pooled_maps, dim=1)
        # 将 concatenated_features 传递给分类器
        output = self.classifier(concatenated_features)
        

        return output, p_layers


def test():
    net = MobileNetV2_dynamicFPN(num_classes = 172)
    print(net)
    output, maps = net(torch.randn(1, 3, 224, 224))
    for feature_map in maps:
        print(feature_map.size())
    print(output.shape)

# tensorboard --logdir=logs\loss_2022_11_24_09_59_09
test()