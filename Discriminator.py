import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_channels, output_channels=64, n_inner_layers=2, norm_layer=nn.BatchNorm2d):
        """ Parameters:
            input_channels (int)  -- the number of channels in input images
            output_channels (int) -- the number of filters that come out of the first layer
            n_inner_layers (int)  -- the number of inner convolutional layers
            norm_layer      -- normalization layer
        """
        super().__init__()
        use_bias = False
        filter_size = 4
        padding_width = 1
        architecture = []

        # First layer, does not apply batchNorm
        architecture += nn.Conv2d(input_channels, output_channels, kernel_size=filter_size, stride=2, padding=padding_width)
        architecture += nn.LeakyReLU(0.2, True)

        # Second and third layer
        multiplier = 1
        for n in range(n_inner_layers):
            architecture += [nn.Conv2d(output_channels * multiplier, output_channels * (multiplier ** 2), kernel_size=filter_size, stride=2,
                                       padding=padding_width, bias=use_bias)]
            architecture += norm_layer(output_channels * (multiplier ** 2))
            architecture += nn.LeakyReLU(0.2, True)
            multiplier *= 2

        # Final layer, outputs a single value, which represents the belief of the discriminator
        architecture += [nn.Conv2d(output_channels * multiplier, 1, kernel_size=filter_size, stride=1, padding=padding_width)]

        self.model = nn.Sequential(*architecture)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)