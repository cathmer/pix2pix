import torch.nn as nn

FIRST_LAYER_FILTERS = 64
INPUT_CHANNELS = 6
N_INNER_LAYERS = 2


class Discriminator(nn.Module):
    def __init__(self):
        """ Parameters:
        """
        super().__init__()
        use_bias = False
        filter_size = 4
        padding_width = 1
        architecture = []

        # First layer, does not apply batchNorm
        architecture += nn.Conv2d(INPUT_CHANNELS, FIRST_LAYER_FILTERS, kernel_size=filter_size, stride=2, padding=padding_width)
        architecture += nn.LeakyReLU(0.2, True)

        # Second and third layer
        multiplier = 1
        for n in range(N_INNER_LAYERS):
            architecture += [nn.Conv2d(FIRST_LAYER_FILTERS * multiplier, FIRST_LAYER_FILTERS * (multiplier ** 2), kernel_size=filter_size, stride=2,
                                       padding=padding_width, bias=use_bias)]
            architecture += nn.BatchNorm2d(FIRST_LAYER_FILTERS * (multiplier ** 2))
            architecture += nn.LeakyReLU(0.2, True)
            multiplier *= 2

        # Final layer, outputs a single value, which represents the belief of the discriminator
        architecture += [nn.Conv2d(FIRST_LAYER_FILTERS * multiplier, 1, kernel_size=filter_size, stride=1, padding=padding_width)]

        self.model = nn.Sequential(*architecture)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)