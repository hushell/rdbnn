import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

model_urls = {'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth'}

class VGG(nn.Module):
    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'G':
            layers += [nn.AdaptiveAvgPool2d(1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

cfg = {
    "vgg4": [64, 'M', 128, 'M', 512, 'M', 'G'],
    "vgg9": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 'G']
    }

def vgg4(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg["vgg4"]), **kwargs)

    model.feature_modules = [
        (num_channel, list(model.features.children())[i]) for (num_channel, i) in [(128, 7)]]

    if pretrained:
        model.load_state_dict(torch.load('./pts/vgg4pt'))

    return model

def vgg9(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg["vgg9"]), **kwargs)

    model.feature_modules = [
        (num_channel, list(model.features.children())[i]) for (num_channel, i) in [
            (128, 7), (256, 14), (512, 21), (512, 28)]]

    if pretrained:
        state_dict = model_zoo.load_url(model_urls['vgg11_bn'])
        state_dict = {k: v for k, v in state_dict.items() if "feature" in k}

        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    return model
