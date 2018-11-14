import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GrayResnet18(models.ResNet):
    def __init__(self, dimOut, dropout):
        super(GrayResnet18, self).__init__(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=1000)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.extraFc1 = nn.Linear(1000, 200)
        self.drop1 = nn.Dropout(p=dropout)
        self.extraFc2 = nn.Linear(200, dimOut)

    def forward(self, x):
        x = super(GrayResnet18, self).forward(x)
        x = F.relu(self.extraFc1(x))
        x = self.drop1(x)
        x = self.extraFc2(x)
        return F.log_softmax(x, dim=1)
