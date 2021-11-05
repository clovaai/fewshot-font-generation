"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(512 * 4, output_dim)

    def forward(self, x):
        if x.size(1) == 1:
            x = torch.cat([x, x, x], dim=1)
        out = self.model(x)

        return out
