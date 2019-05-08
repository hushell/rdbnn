###
# Code modified from https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
###

import torch
import math

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from models.modules import conv1x1, conv3x3

class BasicBlock(nn.Module):
    # basic block for building groupped blocks
    def __init__(self, in_channels, out_channels, stride, drop_rate):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, 1)

        self.is_equal_in_out = (in_channels==out_channels)
        if not self.is_equal_in_out:
            # use shortcut if in_channel != out_channels
            self.conv_shortcut = conv1x1(in_channels, out_channels, stride)

        self.drop_rate = drop_rate

    def forward(self, x):
        if self.is_equal_in_out:
            out = self.conv1(self.relu1(self.bn1(x)))
            out = self.relu2(self.bn2(out))
            if self.drop_rate > 0:
                out = F.dropout(out, p=self.drop_rate, training=self.training)

            out = self.conv2(out)
            out = torch.add(out, x)

        else:
            x = self.relu1(self.bn1(x)) # retain x for conv_shortcut later
            out = self.conv1(x)
            out = self.relu2(self.bn2(out))
            if self.drop_rate > 0:
                out = F.dropout(out, p=self.drop_rate, training=self.training)

            out = self.conv2(out)
            out = torch.add(out, self.conv_shortcut(x))

        return out

class GroupBlock(nn.Module):
    # groupped blocks for building resnets
    def __init__(self, num_blocks, in_channels, out_channels, stride, drop_rate):
        super(GroupBlock, self).__init__()

        self.layer = self._make_layer(
            in_channels, out_channels, num_blocks, stride, drop_rate)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, drop_rate):
        layers = []
        for i in range(int(num_blocks)):
            if i == 0:
                # in_channels x out_channels mapping with stride if first layer
                _in_channels, _out_channels, _stride = (
                    in_channels, out_channels, stride)
            else:
                # out_channels x out_channels mapping if not first layer
                _in_channels, _out_channels, _stride = (
                    out_channels, out_channels, 1)

            layers.append(BasicBlock(_in_channels, _out_channels, _stride, drop_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    # wide residual network, code modified from
    # depth of network is determined by number of layers;
    # 4 layers at top and 6 layers per blocks each.
    # width of network determines the size of channels for each groups

    def __init__(self, depth, width, num_classes=10, drop_rate=0.0):
        super(WideResNet, self).__init__()
        self.depth = depth
        self.width = width
        self.layers = [16*width, 32*width, 64*width]
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        # each blocks needs to be built from 6 layers each
        assert((self.depth-4)%6 == 0)

        self.num_blocks_per_group = (self.depth-4)/6

        # 1st conv before any group
        self.conv1 = conv3x3(3, 16, 1)

        # 1st group
        self.layer1 = GroupBlock(
            num_blocks=self.num_blocks_per_group,
            in_channels=16,
            out_channels=self.layers[0],
            stride=1,
            drop_rate=self.drop_rate)

        # 2nd group
        self.layer2 = GroupBlock(
            num_blocks=self.num_blocks_per_group,
            in_channels=self.layers[0],
            out_channels=self.layers[1],
            stride=2,
            drop_rate=self.drop_rate)

        # 3rd group
        self.layer3 = GroupBlock(
            num_blocks=self.num_blocks_per_group,
            in_channels=self.layers[1],
            out_channels=self.layers[2],
            stride=2,
            drop_rate=self.drop_rate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(self.layers[2])
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(self.layers[2], self.num_classes)
        self.view_dim = self.layers[2]

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]* m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))
        out = self.avg_pool2d(out)
        out = out.view(-1, self.view_dim)
        out = self.fc(out)

        return out


def wideresnet_16_1(pretrained=False, ckpt_pth='./pths/wideresnet_16_1.pth', **kwargs):
    """Constructs a WideResNet-16-1 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on CIFAR10
    """
    model = WideResNet(16, 1, **kwargs)
    model.feature_modules = [model.layer1, model.layer2, model.layer3]

    if pretrained:
        model.load_state_dict(torch.load(ckpt_pth))

    return model

def wideresnet_16_2(pretrained=False, ckpt_pth='./pths/wideresnet_16_2.pth', **kwargs):
    """Constructs a WideResNet-16-2 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on CIFAR10
    """
    model = WideResNet(16, 2, **kwargs)
    model.feature_modules = [model.layer1, model.layer2, model.layer3]

    if pretrained:
        model.load_state_dict(torch.load(ckpt_pth))

    return model

def wideresnet_40_1(pretrained=False, ckpt_pth='./pths/wideresnet_40_1.pth', **kwargs):
    """Constructs a WideResNet-40-1 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on CIFAR10
    """
    model = WideResNet(40, 1, **kwargs)
    model.feature_modules = [model.layer1, model.layer2, model.layer3]

    if pretrained:
        model.load_state_dict(torch.load(ckpt_pth))

    return model

def wideresnet_40_2(pretrained=False, ckpt_pth='./pths/wideresnet_40_2.pth', **kwargs):
    """Constructs a WideResNet-40-2 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on CIFAR10
    """
    model = WideResNet(40, 2, **kwargs)
    model.feature_modules = [model.layer1, model.layer2, model.layer3]

    if pretrained:
        model.load_state_dict(torch.load(ckpt_pth))

    return model

def wideresnet_40_4(pretrained=False, ckpt_pth='./pths/wideresnet_40_4.pth', **kwargs):
    """Constructs a WideResNet-40-2 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on CIFAR10
    """
    model = WideResNet(40, 4, **kwargs)
    model.feature_modules = [model.layer1, model.layer2, model.layer3]

    if pretrained:
        model.load_state_dict(torch.load(ckpt_pth))

    return model
