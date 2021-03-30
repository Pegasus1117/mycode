import torch
import torch.nn as nn
from torchvision import models

def pooling():
    m = nn.AdaptiveMaxPool2d((5, 7))
    input = torch.randn(1, 64, 8, 9)
    output = m(input)

    m = nn.AdaptiveMaxPool2d(3)
    input = torch.randn(1, 3, 10, 9)
    output = m(input)

    print(input.size())
    print(output.size())
    print(output)
    print(output * output)


def printModules():
    prop_feats = nn.Sequential(
        nn.Conv2d(2048, 512, 3, padding=1),
        nn.ReLU(inplace=True),
    )
    print(prop_feats)
    print(prop_feats[0].out_channels)



if __name__ == "__main__":
    pooling()
    printModules()