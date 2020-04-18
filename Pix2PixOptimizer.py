import torch
import torch.nn as nn

from Discriminator import Discriminator
from Generator import UNetGenerator
from torch import Tensor

LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
LAMBDA = 100


class Pix2PixOptimizer:
    def __init__(self, is_train=True, use_GAN=True, is_conditional=True, has_L1=True, use_cuda=False):
        self.use_GAN = use_GAN
        self.is_conditional = is_conditional
        self.has_L1 = has_L1
        self.use_cuda = use_cuda and torch.cuda.is_available()

        # Create Generator network
        if self.use_cuda:
            self.generator = UNetGenerator().cuda()
        else:
            self.generator = UNetGenerator()

        # If this is training, a discriminator and optimizers need to be initialized
        if is_train:
            # Create Discriminator network
            if self.is_conditional:
                self.discriminator = Discriminator(6)
            else:
                self.discriminator = Discriminator(3)

            # Set GANLoss to Mean Squared Error
            self.GANLoss = nn.MSELoss()
            self.L1Loss = nn.L1Loss()

            # Apply cuda
            if self.use_cuda:
                self.discriminator.cuda()
                self.GANLoss.cuda()
                self.L1Loss.cuda()

            # Adam optimization
            self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=LR, betas=(BETA1, BETA2))
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))

    def forward(self, real_A):
        # The forward pass, passes a cityscape image to the Generator network which should generate a city image from it
        if self.use_cuda:
            fake_B = self.generator.forward(real_A.cuda())
        else:
            fake_B = self.generator.forward(real_A)
        return fake_B

    def backward_d_AB(self, A, B, target: bool, detach: bool):
        if self.is_conditional:
            # For conditional GAN we want to give both the input image and the generated image to the network
            input_image = torch.cat((A, B), 1)
        else:
            input_image = B

        # Detach the generated image to prevent back propagation to the generator
        if detach:
            input_image = input_image.detach()

        # Get prediction from discriminator on the generated image
        if self.use_cuda:
            prediction = self.discriminator(input_image.cuda())
        else:
            prediction = self.discriminator(input_image)

        # Create a target tensor
        if target:
            target_tensor = torch.tensor(1.0).requires_grad_(False).expand_as(prediction)
        else:
            target_tensor = torch.tensor(0.0).requires_grad_(False).expand_as(prediction)

        # Calculate the GANLoss
        if self.use_cuda:
            loss = self.GANLoss(prediction.cuda(), target_tensor.cuda())
        else:
            loss = self.GANLoss(prediction, target_tensor)
        return loss

    def backward_discriminator(self, real_A, real_B, fake_B):
        # Train on fake image
        loss_fake = self.backward_d_AB(real_A, fake_B, False, True)

        # Train on real
        loss_real = self.backward_d_AB(real_A, real_B, True, False)

        # Combine the losses calculated above
        loss = 0.5 * (loss_fake + loss_real)

        # Backward propagate the losses
        loss.backward()

        return loss

    def backward_generator(self, real_A, real_B, fake_B) -> Tensor:
        # If using GAN, put generated image into Discriminator network.
        if self.use_GAN:
            # Train on fake image to fool the discriminator
            loss = self.backward_d_AB(real_A, fake_B, True, False)

        if self.has_L1:
            # Calculate the L1 Loss between the generated image and the real image.
            if self.use_cuda:
                loss_L1 = self.L1Loss(fake_B.cuda(), real_B.cuda()) * LAMBDA
            else:
                loss_L1 = self.L1Loss(fake_B, real_B) * LAMBDA

            if self.use_GAN:
                loss = loss + loss_L1
            else:
                loss = loss_L1

        # Backward propagate the losses
        loss.backward()

        return loss

    def optimize(self, images):
        if self.use_cuda:
            real_A = images['A'].cuda()
            real_B = images['B'].cuda()
        else:
            real_A = images['A']
            real_B = images['B']

        discriminator_loss = 0.0

        # Generate fake image from a semantic image
        fake_B = self.forward(real_A)

        # Update the discriminator if GAN is used
        if self.use_GAN:
            # Enable backpropagation for the discriminator
            self.discriminator.set_requires_grad(True)
            # Set the gradients to zero
            self.discriminator_optimizer.zero_grad()
            # Calculate the new gradients
            discriminator_loss = self.backward_discriminator(real_A, real_B, fake_B)
            # Update weights
            self.discriminator_optimizer.step()

        # Update Generator
        # Don't calculate gradients for D when optimizing G
        self.discriminator.set_requires_grad(False)
        # Set gradients to zero
        self.generator_optimizer.zero_grad()
        # Calculate gradients
        generator_loss = self.backward_generator(real_A, real_B, fake_B)
        # Update weights
        self.generator_optimizer.step()

        # Return the error
        return generator_loss, discriminator_loss, fake_B
