from torchvision import models
from torch import nn
import torch.nn.functional as F
from fnmatch import fnmatch
from torch import tensor
import torch
import numpy as np


class PretrainedResnetFrozen(nn.Module):
    def __init__(self, dim_out, min_norm, max_norm, mean_norm, std_norm):
        super().__init__()
        self.min_norm = torch.as_tensor(min_norm.astype(np.float)).cuda()
        self.max_norm = torch.as_tensor(max_norm.astype(np.float)).cuda()
        self.mean_norm = torch.as_tensor(mean_norm.astype(np.float)).cuda()
        self.std_norm = torch.as_tensor(std_norm.astype(np.float)).cuda()
        self.channel_mean = tensor([0.485, 0.456, 0.406]).cuda().reshape(1, -1, 1, 1)
        self.channel_std = tensor([0.229, 0.224, 0.225]).cuda().reshape(1, -1, 1, 1)
        self.ResNet = models.resnet18(pretrained=True)
        # pytorch's standard implementation throws errors at some image sizes..
        self.ResNet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ResNet.fc = nn.Linear(512, dim_out)
        self.freeze_except_fc()

    def forward(self, x):
        """
        We are using pretrained weights here. So the best way to use them, is to multiply the grayscale
        image to rbg. After that, it is best to normalize the image the way, the images were normalized.
        To do so, we first squeeze the values between 0 and 1 and then transform the way, the resnet images
        were transformed.
        Normally, this functionality is put into the dataloader, but I want to keep the dataloader general.

        :param x:
        :return:
        """
        # squeeze values between 0 and 1. Use values from test_data instead of small batch to decrease statistical variance
        # If you think about the math, this does the trick (max_norm and min_norm are from the pre-normalized distribution)
        x = x*(self.std_norm/(self.max_norm-self.min_norm)) + (self.mean_norm-self.min_norm)/(self.max_norm - self.min_norm)
        # copy to 3 channels
        x = x.repeat(1, 3, 1, 1)
        # substract imagenet mean and scale imagenet std
        x -= self.channel_mean
        x /= self.channel_std
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
    Net = PretrainedResnetFrozen(2)
    for name, param in Net.named_parameters():
        if param.requires_grad:
            print(name)
    # Net.unfreeze_all()
    for name, param in Net.named_parameters():
        if param.requires_grad:
            print(name)
