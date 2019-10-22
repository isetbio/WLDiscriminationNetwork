import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class vgg16(nn.Module):
    def __init__(self, dimOut):
        super().__init__()
        self.model = models.vgg16(num_classes=dimOut)

        # self.extraFc1 = nn.Linear(1000, 200)
        # self.drop1 = nn.Dropout(p=dropout)
        # self.extraFc2 = nn.Linear(200, dimOut)

    def forward(self, x):
        # copy to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.model.forward(x)
        # x = F.relu(self.extraFc1(x))
        # x = self.drop1(x)
        # x = self.extraFc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    test = vgg16(2)
    print('nice')