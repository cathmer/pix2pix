import torch.nn as nn

import torch.nn.init as weight_init

FIRST_LAYER_FILTERS = 64
N_INNER_LAYERS = 2


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        # TODO: Set Parameter for number of input channels (can be 3 for GAN, 6 for cGAN)
        """ Parameters:
        """
        super().__init__()
        use_bias = False
        filter_size = 4
        padding_width = 1
        architecture = []

        mean = 0.0
        std = 0.02

        # First layer, does not apply batchNorm
        layer1 = nn.Conv2d(input_channels, FIRST_LAYER_FILTERS, kernel_size=filter_size, stride=2,
                           padding=padding_width)
        weight_init.normal_(layer1.weight, mean=mean, std=std)
        architecture += [layer1]
        architecture += [nn.LeakyReLU(0.2, True)]

        # Second and third layer
        multiplier = 1
        for n in range(N_INNER_LAYERS):
            layer2 = nn.Conv2d(FIRST_LAYER_FILTERS * multiplier, FIRST_LAYER_FILTERS * multiplier * 2,
                               kernel_size=filter_size, stride=2, padding=padding_width, bias=use_bias)
            weight_init.normal_(layer2.weight, mean=mean, std=std)
            architecture += [layer2]

            layer3 = nn.BatchNorm2d(FIRST_LAYER_FILTERS * multiplier * 2)
            weight_init.normal_(layer3.weight, mean=mean, std=std)
            architecture += [layer3]
            architecture += [nn.LeakyReLU(0.2, True)]
            multiplier *= 2

        # Final layer, outputs a single value, which represents the belief of the discriminator
        layer4 = nn.Conv2d(FIRST_LAYER_FILTERS * multiplier, 1, kernel_size=filter_size, stride=1,
                           padding=padding_width)
        weight_init.normal_(layer4.weight, mean=mean, std=std)
        architecture += [layer4]

        self.model = nn.Sequential(*architecture)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)
