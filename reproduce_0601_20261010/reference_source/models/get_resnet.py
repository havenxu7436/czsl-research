import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet, BasicBlock


class ResNet18_conv(ResNet):
    def __init__(self):
        super(ResNet18_conv, self).__init__(BasicBlock, [2, 2, 2, 2])

    def forward(self, x):
        # change forward here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        multilayer = []
        x = self.layer1(x)
        multilayer.append(x)
        x = self.layer2(x)
        multilayer.append(x)
        x = self.layer3(x)
        multilayer.append(x)
        x = self.layer4(x)
        multilayer.append(x)

        return multilayer

def get_image_extractor(arch='resnet18', pretrained = True, feature_dim = None, checkpoint = ''):
    '''
    Inputs
        arch: Base architecture
        pretrained: Bool, Imagenet weights
        feature_dim: Int, output feature dimension
        checkpoint: String, not implemented
    Returns
        Pytorch model
    '''

    if arch == 'resnet18':
        model = models.resnet18(pretrained = pretrained)
        if feature_dim is None:
            model.fc = nn.Sequential()
        else:
            model.fc = nn.Linear(512, feature_dim)

    if arch == 'resnet18_conv':
        model = ResNet18_conv()
        model.load_state_dict(models.resnet18(pretrained=True).state_dict())

    return model

