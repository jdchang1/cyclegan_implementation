import torch
import torch.nn as nn
import itertools

from utils import LearningRateDecay
from generator import ResNetGenerator
from discriminator import Discriminator

class CycleGan():
    def __init__(self, in_c, out_c, is_train=True, lr=0.0002, num_epoch=200, offset=0, decay_start=100, cuda=True):
        self.G_A2B = ResNetGenerator(in_c, out_c)
        self.G_B2A = ResNetGenerator(out_c, in_c)
        self.D_A = Discriminator(in_c)
        self.D_B = Discriminator(out_c)

        if cuda:
            self.G_A2B.cuda()
            self.G_B2A.cuda()
            self.D_A.cuda()
            self.D_B.cuda()

        # Init convolutional layers from a normal distribution (mu-0, sigma-0.02)
        self.initialize_weights()

        # Define criterions
        self.idt_criterion = nn.L1Loss()
        self.cycle_criterion = nn.L1Loss()
        self.gan_criterion = nn.MSELoss()

        # Define optimizers and learning rate schedulers if training
        if is_train:
            self.initialize_optim(lr, num_epoch, offset, decay_start)


    def initialize_weights(self):
        def normal_init(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)

        self.G_A2B.apply(normal_init)
        self.G_B2A.apply(normal_init)
        self.D_A.apply(normal_init)
        self.D_B.apply(normal_init)

    def initialize_optim(self, lr, num_epoch, offset, decay_start):
        # Optimizers
        self.optim_g = torch.optim.Adam(itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
        self.optim_d = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=lr, betas=(0.5, 0.999))

        # Schedulers
        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim_g, lr_lambda=LearningRateDecay(num_epoch, offset, decay_start).step)
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim_d, lr_lambda=LearningRateDecay(num_epoch, offset, decay_start).step)

    def lr_update(self):
        self.g_scheduler.step()
        self.d_scheduler.step()

