import torch
import torch.nn as nn
from Generator import UNetGenerator
from Discriminator import Discriminator

LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

'''
TODO'S:
- Implement backward pass for discriminator and generator
- Implement optimize function
- Preprocess input images
- Set up evaluation function
- Set up training
- Set up testing
'''


class Pix2PixOptimizer():
    def __init__(self, input_images, is_train=True, is_conditional=True, has_L1=True):
        # TODO: SET SELF.DEVICE AND PUT GANLOSS AND INPUT IMAGES TO THIS DEVICE

        # Create Generator network
        self.Gnet = UNetGenerator()

        # Set input images
        self.real_A = input_images['A']

        # If has_L1 is true, the optimizer is trained with an L1 loss
        if has_L1:
            self.lamb = 100
        else:
            self.lamb = 0

        # If this is training, a discriminator and optimizers need to be initialized
        if is_train:
            # Create Discriminator network
            self.Dnet = Discriminator()
            # Set GANLoss to Mean Squared Error
            self.GANLoss = nn.MSELoss()
            self.L1Loss = nn.L1Loss()

            # Adam optimization
            self.G_optimizer = torch.optim.Adam(self.Gnet.parameters(), lr=LR, betas=(BETA1, BETA2))
            self.D_optimizer = torch.optim.Adam(self.Dnet.parameters(), lr=LR, betas=(BETA1, BETA2))
            self.optimizers.append(self.G_optimizer)
            self.optimizers.append(self.D_optimizer)

    def forward(self):
        # The forward pass, passes a cityscape image to the Generator network which should generate a city image from it
        self.generated_B = self.Gnet(self.real_A)

    def backward_d(self):
        pass

    def backward_g(self):
        pass

    def optimize(self):
        pass
