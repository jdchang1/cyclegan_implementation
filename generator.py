import torch.nn as nn

class ResBlock(nn.Module):
    """
    Residual blocks used in the generator architecture
    """
    def __init__(self, in_c):
        """
        Args:
            in_c (int): number of input features for the block
        """
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(*[nn.ReflectionPad2d(1),
                                         nn.Conv2d(in_c, in_c, 3),
                                         nn.InstanceNorm2d(in_c),
                                         nn.ReLU(inplace=True),
                                         nn.ReflectionPad2d(1),
                                         nn.Conv2d(in_c, in_c, 3),
                                         nn.InstanceNorm2d(in_c)])
    def forward(self, input):
        return input + self.res_block(input)

class ConvBlock(nn.Module):
    """
    The convolution blocks that are the first and last blocks of the generator
    """
    def __init__(self, in_c, out_c, normalize=True, act_func='relu'):
        """
        Args:
            in_c (int): number of input features
            out_c (int): number of output features
            normalize (Boolean): True for instance norm, False otherwise
            act_func (String): Activation function name (relu or tanh)
        """
        super(ConvBlock, self).__init__()
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_c, out_c, 7)]
        if normalize:
            layers += [nn.InstanceNorm2d(out_c)]
        if act_func == 'relu':
            layers += [nn.ReLU(inplace=True)]
        elif act_func == 'tanh':
            layers += [nn.Tanh()]
        else:
            raise ValueError('Not a valid activation function for this architecture: relu or tanh')
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv_block(input)

class DownsampleBlock(nn.Module):
    """
    The convolution blocks used for downsampling the input
    """
    def __init__(self, in_c, out_c):
        super(DownsampleBlock, self).__init__()
        self.dk = nn.Sequential(*[nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
                                  nn.InstanceNorm2d(out_c),
                                  nn.ReLU(inplace=True)])
    def forward(self, input):
        return self.dk(input)

class UpsamplingBlock(nn.Module):
    """
    The convolution with fractional padding or deconvolution blocks used for upsampling the output
    """
    def __init__(self, in_c, out_c):
        super(UpsamplingBlock, self).__init__()
        self.uk = nn.Sequential(*[nn.ConvTranspose2d(in_c, out_c, 3, stride=2, padding=1, output_padding=1),
                                  nn.InstanceNorm2d(out_c),
                                  nn.ReLU(inplace=True)])

    def forward(self, input):
        return self.uk(input)

class ResNetGenerator(nn.Module):
    def __init__(self, in_c, out_c, n_res=9):
        super(ResNetGenerator, self).__init__()
        self.model = nn.Sequential()
        # Downsample
        self.model.add_module('ck1', ConvBlock(in_c, 64))
        self.model.add_module('dk1', DownsampleBlock(64, 128))
        self.model.add_module('dk2', DownsampleBlock(128, 256))
        
        # Residual Blocks
        for i in range(n_res):
            self.model.add_module('r{}'.format(i), ResBlock(256))
        
        # Upsample
        self.model.add_module('uk1', UpsamplingBlock(256, 128))
        self.model.add_module('uk2', UpsamplingBlock(128, 64))
        self.model.add_module('ck2', ConvBlock(64, out_c, normalize=False, act_func='tanh'))

    def forward(self, input):
        return self.model(input)
