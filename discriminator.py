import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, normalize=True):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)]
        if normalize:
            layers += [nn.InstanceNorm2d(out_c)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.conv_block(input)

class Discriminator(nn.Module):
    def __init__(self, in_c):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(*[ConvBlock(in_c, 64, normalize=False),
                                     ConvBlock(64, 128),
                                     ConvBlock(128, 256),
                                     ConvBlock(256, 512),
                                     nn.Conv2d(512, 1, 4, padding=1)])
    def forward(self, input):
        return self.model(input)

