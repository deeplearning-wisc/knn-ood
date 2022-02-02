import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet50', ]


normalization = nn.BatchNorm2d

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Identity(nn.Module):
    def forward(self, input):
        return input + 0.0

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = normalization(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = normalization(planes)
        self.shortcut = Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.shortcut(out)
        out = self.relu(out)

        return out

    def forward_masked(self, x, mask_weight=None, mask_bias=None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.shortcut(out)
        out = self.relu(out)

        if mask_weight is not None:
            out = out * mask_weight[None,:,None,None]
        if mask_bias is not None:
            out = out + mask_bias[None,:,None,None]
        return out

    def forward_threshold(self, x, threshold=1e10):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        b, c, w, h = out.shape
        mask = out.view(b, c, -1).mean(2) < threshold
        out = mask[:, :, None, None] * out
        # print(mask.sum(1).float().mean(0))
        return out


class WideBasicBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(WideBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = normalization(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes * 4)
        self.bn2 = normalization(planes * 4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

    def forward_masked(self, x, mask=None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)


        out = out + residual
        if mask is not None:
            out = out * mask[None,:,None,None]# + self.bn2.bias[None,:,None,None] * (1 - mask[None,:,None,None])

        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normalization(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = normalization(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = normalization(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.shortcut = Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.shortcut(out)
        out = self.relu(out)

        return out

    def forward_masked(self, x, mask_weight=None, mask_bias=None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        if mask_weight is not None:
            out = out * mask_weight[None,:,None,None]
        if mask_bias is not None:
            out = out + mask_bias[None,:,None,None]
        out = self.relu(out)
        return out

    def forward_threshold(self, x, threshold=1e10):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        b, c, w, h = out.shape
        mask = out.view(b, c, -1).mean(2) < threshold
        out = mask[:, :, None, None] * out

        return out


class AbstractResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(AbstractResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = normalization(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _initial_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                normalization(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)

        if strict:
            error_msg = ''
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            print('Warning(s) in loading state_dict for {}:\n\t{}'.format(self.__class__.__name__, "\n\t".join(error_msgs)))


class ResNet(AbstractResNet):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._initial_weight()

    def forward_masked(self, x, mask_weight=None, mask_bias=None):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.layer4[:-1](x)
        x = self.layer4[-1].forward_masked(x, mask_weight=mask_weight, mask_bias=mask_bias)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward_threshold(self, x, threshold=1e10):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        # x = self.layer3(self.layer2(self.layer1(x)))
        # x = self.layer4[:-1](x)
        # x = self.layer4[-1].forward_threshold(x, threshold=1e10)
        x= self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = x.clip(max=threshold)
        # mask = x < threshold
        # mask = mask.float()
        # x = mask * x + (1-mask.float()) * (2.)
        x = x.view(x.size(0), -1)

        # if self.fc.weight.data.min().item() < 0:
        #     w = self.fc.weight.data
        #     w = w - self.fc.weight.data.min()
        #     self.fc.weight.data = w
        x = self.fc(x)
        return x

    def feature_list(self, x):
        out_list = []
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = self.avgpool(out)
        # out = out.clip(max=1.0)
        # out_list.append(out)
        out = out.view(out.size(0), -1)
        y = self.fc(out)
        return y, out_list

    def intermediate_forward(self, x, layer_index):
    # if layer_index >= 0:
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
    # if layer_index >= 1:
        out = self.layer1(out)
    # if layer_index >= 2:
        out = self.layer2(out)
    # if layer_index >= 3:
        out = self.layer3(out)
    # if layer_index >= 4:
        out = self.layer4(out)
        out = self.avgpool(out)
        # out = out.clip(max=1.0)
        return out


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model




class ResNetCifar(AbstractResNet):
    def __init__(self, block, layers, num_classes=10, method='', p=None, info=None):
        super(ResNetCifar, self).__init__(block, layers, num_classes)
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.method = method

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if method.find("ood") > -1:
            self.fc_ood = RouteFcMaxAct(512 * block.expansion, 1, topk=p)

        self.avgpool = nn.AvgPool2d(4, stride=1)
        self._initial_weight()

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def forward(self, x, fc_params=None):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return out

    def forward_masked(self, x, mask_weight=None, mask_bias=None):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.layer4[:-1](x)
        x = self.layer4[-1].forward_masked(x, mask_weight=mask_weight, mask_bias=mask_bias)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward_threshold(self, x, threshold=1e10):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = self.layer3(self.layer2(self.layer1(x)))
        # x = self.layer4[:-1](x)
        # x = self.layer4[-1].forward_threshold(x, threshold=threshold)
        x= self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = x.clip(max=threshold)
        x = x.view(x.size(0), -1)
        return self.fc(x)


    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        y = self.fc(out)
        return y, out_list

    def intermediate_forward(self, x, layer_index):
        # if layer_index >= 0:
        out = F.relu(self.bn1(self.conv1(x)))
        # if layer_index >= 1:
        out = self.layer1(out)
        # if layer_index >= 2:
        out = self.layer2(out)
        # if layer_index >= 3:
        out = self.layer3(out)
        # if layer_index >= 4:
        out = self.layer4(out)
        return out


def resnet18_cifar(**kwargs):
    return ResNetCifar(BasicBlock, [2,2,2,2], **kwargs)


def resnet34_cifar(**kwargs):
    return ResNetCifar(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50_cifar(**kwargs):
    return ResNetCifar(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101_cifar(**kwargs):
    return ResNetCifar(Bottleneck, [3, 4, 23, 3], **kwargs)