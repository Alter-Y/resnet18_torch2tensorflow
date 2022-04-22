import torch
import torch.nn as nn


class Conv1(nn.Module):
    # first layer
    def __init__(self, in_channels, out_channels, ksize=7, stride=2, padding=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=ksize,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.maxpool(self.act(self.bn(self.conv(x))))

class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=ksize,
                              stride=stride,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class Basicblock(nn.Module):
    # basic block
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, downsampling=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=ksize,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=ksize,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = Downsampling(in_channels,
                                     out_channels,
                                     stride=stride)
        self.downsampling = downsampling

    def forward(self, x):
        y = x
        x_1 = self.act(self.bn1(self.conv1(x)))
        x_2 = self.bn2(self.conv2(x_1))

        if self.downsampling:
            y = self.residual(y)

        out = x_2 + y
        out = self.act(out)

        return out

class Resnet(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super().__init__()
        self.conv1 = Conv1(3, 64)
        self.layer1 = self.make_layer(64, 64, layers[0], stride=2)
        self.layer2 = self.make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(256, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_channels, out_channels, layer, stride):
        layers = []
        layers.append(Basicblock(in_channels, out_channels, stride=stride, downsampling=True))
        for _ in range(1,layer):
            layers.append(Basicblock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(num_classes):
    return Resnet([2,2,2,2], num_classes)


if __name__ == '__main__':
    from torchsummary import summary
    model = resnet18(10)
    print(model)

    summary(model, (3,32,32))