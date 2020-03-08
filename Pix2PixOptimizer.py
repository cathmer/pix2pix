import torch
import torch.nn as nn
from Network import UNetGenerator
from Discriminator import Discriminator

LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

class Pix2PixOptimizer():


    def __init__(self, is_train=True):
        self.Gnet = UNetGenerator()

        if is_train:
            self.Dnet = Discriminator()
            self.GANLoss = nn.MSELoss()
            self.L1Loss = nn.L1Loss()

            self.G_optimizer = torch.optim.Adam(self.Gnet.parameters(), lr=LR, betas=(BETA1, BETA2))
            self.D_optimizer = torch.optim.Adam(self.Dnet.parameters(), lr=LR, betas=(BETA1, BETA2))
            self.optimizers.append(self.G_optimizer)
            self.optimizers.append(self.D_optimizer)



