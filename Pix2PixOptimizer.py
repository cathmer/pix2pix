import torch
import torch.nn as nn

from Discriminator import Discriminator
from Generator import UNetGenerator

import torchvision

LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999


class Pix2PixOptimizer:
    def __init__(self, is_train=True, use_GAN=True, is_conditional=True, has_L1=True, use_cuda=False):
        # TODO: SET SELF.DEVICE AND PUT GANLOSS AND INPUT IMAGES TO THIS DEVICE

        # self.save_images = save_images
        self.use_cuda = use_cuda and torch.cuda.is_available()

        # Create Generator network
        self.Gnet = UNetGenerator()
        # Apply cuda
        if self.use_cuda:
            self.Gnet.cuda()

        # If has_L1 is true, the optimizer is trained with an L1 loss
        self.has_L1 = has_L1
        if has_L1:
            self.lamb = 100
        else:
            self.lamb = 0

        self.is_conditional = is_conditional
        self.use_GAN = use_GAN

        # If this is training, a discriminator and optimizers need to be initialized
        if is_train:
            # Create Discriminator network
            if self.is_conditional:
                self.Dnet = Discriminator(6)
            else:
                self.Dnet = Discriminator(3)

            # Set GANLoss to Mean Squared Error
            self.GANLoss = nn.MSELoss()
            self.L1Loss = nn.L1Loss()

            # Apply cuda
            if self.use_cuda:
                self.Dnet.cuda()
                self.GANLoss.cuda()
                self.L1Loss.cuda()

            # Adam optimization
            self.G_optimizer = torch.optim.Adam(self.Gnet.parameters(), lr=LR, betas=(BETA1, BETA2))
            self.D_optimizer = torch.optim.Adam(self.Dnet.parameters(), lr=LR, betas=(BETA1, BETA2))

    def set_input(self, images):
        if self.use_cuda:
            self.real_A = images['A'].cuda()
            self.real_B = images['B'].cuda()
        else:
            self.real_A = images['A']
            self.real_B = images['B']

    def forward(self, return_image=False):
        # The forward pass, passes a cityscape image to the Generator network which should generate a city image from it
        if self.use_cuda:
            self.Gnet.cuda()
            self.generated_B = self.Gnet.forward(self.real_A.cuda()).cuda()
        else:
            self.generated_B = self.Gnet.forward(self.real_A)

        if return_image:
            return self.generated_B

    def backward_d(self):
        if self.is_conditional:
            # This is a conditional GAN, so we want to give both the input image and the generated image to the network
            generatedImg = torch.cat((self.real_A, self.generated_B), 1)
        else:
            generatedImg = self.generated_B

        # Get prediction from disriminator on the generated image
        # Detach the generated image to prevent backpropagation to the generator

        if self.use_cuda and torch.cuda.is_available():
            self.Dnet.cuda()
            pred_generated = self.Dnet(generatedImg.detach()).cuda()
        else:
            pred_generated = self.Dnet(generatedImg.detach())

        # Get the GANLoss, where the target is False (since it is a generated and therefore fake image)
        target_tensor1 = torch.tensor(0.0).requires_grad_(False).expand_as(pred_generated)

        # Apply cuda
        if self.use_cuda and torch.cuda.is_available():
            target_tensor1.cuda()

        if self.use_cuda and torch.cuda.is_available():
            self.loss_D_generated = self.GANLoss(pred_generated.cuda(), target_tensor1.cuda()).cuda()
        else:
            self.loss_D_generated = self.GANLoss(pred_generated, target_tensor1)

        # Apply cuda
        if self.use_cuda and torch.cuda.is_available():
            self.loss_D_generated.cuda()

        # Now input a real image
        if self.is_conditional:
            # This is for conditional GAN, so give both input image and real image to the network
            realImg = torch.cat((self.real_A, self.real_B), 1)
        else:
            realImg = self.real_B

        # Get the prediction of the network for the real image
        pred_real = self.Dnet(realImg)

        # Apply cuda
        if self.use_cuda and torch.cuda.is_available():
            pred_real.cuda()

        # Get the GANLoss, where the target is true (since it is a real image)
        if self.use_cuda and torch.cuda.is_available():
            target_tensor2 = torch.tensor(1.0).requires_grad_(False).expand_as(pred_real).cuda()
            self.loss_D_real = self.GANLoss(pred_real, target_tensor2).cuda()
        else:
            target_tensor2 = torch.tensor(1.0).requires_grad_(False).expand_as(pred_real)
            self.loss_D_real = self.GANLoss(pred_real, target_tensor2)

        # Combine the losses calculated above
        self.loss_D = 0.25 * (self.loss_D_generated + self.loss_D_real)
        # Backward propagate the losses

        # Apply cuda
        if self.use_cuda and torch.cuda.is_available():
            generatedImg.cuda()
            pred_generated.cuda()
            target_tensor1.cuda()
            self.loss_D_generated.cuda()
            realImg.cuda()
            pred_real.cuda()
            target_tensor2.cuda()
            self.loss_D_real.cuda()
            self.loss_D.cuda()

        self.loss_D.backward()

    def backward_g(self):
        # If using GAN, put generated image into Discriminator network.
        # Else, this is not necessary
        if self.use_GAN:
            if self.is_conditional:
                # Conditional GAN, so concatenate input image and generated image
                generatedImg = torch.cat((self.real_A, self.generated_B), 1)
            else:
                generatedImg = self.generated_B

            # Input the generated image into the discriminator
            pred_generated = self.Dnet(generatedImg)
            # Calculate the GAN Loss; since the Generator wants to fool the Discriminator, the target is reversed here
            # (i.e. the target is True, indicating a real image, although the input is a generated image)
            target_tensor = torch.tensor(1.0).requires_grad_(False).expand_as(pred_generated)

            # Apply cuda
            if self.use_cuda and torch.cuda.is_available():
                self.loss_G_GAN = self.GANLoss(pred_generated.cuda(), target_tensor.cuda()).cuda()
            else:
                self.loss_G_GAN = self.GANLoss(pred_generated, target_tensor)

        # Now calculate the L1 Loss between the generated image and the real image
        if self.has_L1:
            self.loss_G_L1 = self.L1Loss(self.generated_B, self.real_B) * self.lamb

        # Combine the losses
        if self.use_GAN and self.has_L1:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        elif self.use_GAN:
            self.loss_G = self.loss_G_GAN
        else:
            # If not using GAN, simply use L1 loss
            self.loss_G = self.loss_G_L1

        # Apply cuda
        if self.use_cuda and torch.cuda.is_available():
            self.loss_G.cuda()
            if self.has_L1:
                self.loss_G_L1.cuda()

        # Backward propagate the losses
        self.loss_G.backward()

    def optimize(self):
        # Call forward, which puts a semantic image into the Generator which then creates an image from it
        self.forward()

        # Only need to update discriminator if a GAN is used
        if self.use_GAN:
            # Update Discriminator
            # Enable backpropagation for the D
            self.set_requires_grad(self.Dnet, True)
            # Set gradients to zero
            self.D_optimizer.zero_grad()
            # Calculate gradients
            self.backward_d()
            # Update weights
            self.D_optimizer.step()

        # Update Generator
        # Don't calculate gradients for D when optimizing G
        self.set_requires_grad(self.Dnet, False)
        # Set gradients to zero
        self.G_optimizer.zero_grad()
        # Calculate gradients
        self.backward_g()
        # Update weights
        self.G_optimizer.step()

        # Return the error
        return self.loss_G, self.loss_D

    def set_requires_grad(self, netD, requires_grad):
        for param in netD.parameters():
            param.requires_grad = requires_grad
