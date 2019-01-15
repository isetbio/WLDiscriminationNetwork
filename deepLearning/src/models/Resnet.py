from torchvision import models
from torch import nn
import torch.nn.functional as F
from fnmatch import fnmatch


class PretrainedResnetFrozen(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        self.ResNet = models.resnet18(pretrained=True)
        self.ResNet.fc = nn.Linear(512, dim_out)
        self.freeze_except_fc()

    def forward(self, x):
        x = self.ResNet(x)
        return F.log_softmax(x)

    def freeze_except_fc(self):
        for name, param in self.named_parameters():
            if fnmatch(name, '*fc.*'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    Net = PretrainedResnet(2)
    for name, param in Net.named_parameters():
        if param.requires_grad:
            print(name)
    # Net.unfreeze_all()
    for name, param in Net.named_parameters():
        if param.requires_grad:
            print(name)
