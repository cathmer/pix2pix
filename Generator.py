import torch
import torch.nn as nn

import torch.nn.init as weight_init

FIRST_LAYER_FILTERS = 64
INPUT_OUTPUT_CHANNELS = 3


class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        n = FIRST_LAYER_FILTERS

        # Builds the network recursively from inside out
        block = InnermostBlock(n * 8, n * 8)
        for i in range(3):
            block = RegularBlock(n * 8, n * 8, block, use_dropout=True)
        block = RegularBlock(n * 4, n * 8, block)
        block = RegularBlock(n * 2, n * 4, block)
        block = RegularBlock(n, n * 2, block)
        self.model = OutermostBlock(INPUT_OUTPUT_CHANNELS, n, block)

    def forward(self, x):
        return self.model(x)


class UNetBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        mean = 0.0
        std = 0.02

        # This is the "regular" convolution, extracting features and making the image half as big through
        # the stride combined with kernel size and padding
        self.encodeconv = nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1)
        weight_init.normal_(self.encodeconv.weight, mean=mean, std=std)

        # Function for batch normalization, applied at every layer. The encode normalization has output_channels
        # because it is applied after encode convolution.
        self.encodenorm = nn.BatchNorm2d(output_channels)
        weight_init.normal_(self.encodenorm.weight, mean=mean, std=std)
        # The decodenorm has input_channels as argument because it is applied after the decode convolution,
        # which has input_channels as output.
        self.decodenorm = nn.BatchNorm2d(input_channels)
        weight_init.normal_(self.decodenorm.weight, mean=mean, std=std)

        # As stated in the paper, leaky Relu with slope 0.2 is used for encoding
        self.encoderelu = nn.LeakyReLU(0.2, True)

        # For decoding, ReLU without leaking is used
        self.decoderelu = nn.ReLU(True)


class RegularBlock(UNetBlock):
    def __init__(self, input_channels, output_channels, inner_model, use_dropout=False):
        super().__init__(input_channels, output_channels)

        # Decode convolution for a regular block. The number of input channels is 2 times the number of output
        # channels of the associated block because of the skip connection. The number of output channels is
        # equal to the number of input channels of the corresponding block because of the symmetric nature of the
        # network.
        decodeconv = nn.ConvTranspose2d(output_channels * 2, input_channels, kernel_size=4, stride=2, padding=1)

        # Encode does Relu, then Convolution, the Normalization
        encode = [self.encodeconv, self.encodenorm, self.encoderelu]
        # Decode does Relu, then reverse Convolution, the Normalization
        if use_dropout:
            decode = [decodeconv, self.decodenorm, nn.Dropout(0.5), self.decoderelu]
        else:
            decode = [decodeconv, self.decodenorm, self.decoderelu]

        # Combine the blocks with all blocks in between
        model = encode + [inner_model] + decode
        self.model = nn.Sequential(*model)

    # Define skip connections. The block will pass the input forward directly (x) and also pass it through
    # the model (self.model(x)).
    def forward(self, x):
        return torch.cat([x, self.model(x)], 1)


class OutermostBlock(UNetBlock):
    def __init__(self, input_channels, output_channels, inner_model):
        super().__init__(input_channels, output_channels)

        # Because of the skip connection, the decode convolution has double the number of filters as the corresponding
        # encode convolution.
        decodeconv = nn.ConvTranspose2d(output_channels * 2, input_channels, kernel_size=4, stride=2, padding=1)

        # This is the start of the network, so first only Convolution is applied to the input image
        encode = [self.encodeconv, self.encoderelu]

        # Decode, which uses regular ReLU, then convolution to 3 channels (output image) and finally a TanH function as
        # specified in the paper
        decode = [decodeconv, nn.Tanh()]

        # Combine the blocks with all the blocks in between to make a Sequential model
        model = encode + [inner_model] + decode
        self.model = nn.Sequential(*model)

    # No skip connection from the first to the last block, because this would simply be passing the input image to
    # the output
    def forward(self, x):
        return self.model(x)


class InnermostBlock(UNetBlock):
    def __init__(self, input_channels, output_channels):
        super().__init__(input_channels, output_channels)

        # In the innermostblock, the input and output for decoding is simply reversed. No skip connection here so
        # no need to take twice the number of output channels.
        decodeconv = nn.ConvTranspose2d(output_channels, input_channels, kernel_size=4, stride=2, padding=1)

        # Encoding innermost does not apply normalization.
        encode = [self.encodeconv, self.encodenorm, self.encoderelu]
        # Decoding does apply normalization after relu and convolution
        decode = [decodeconv, self.decodenorm, self.decoderelu]

        # No blocks in between so the blocks can simply be put in order in the model
        model = encode + decode
        self.model = nn.Sequential(*model)

    # Pass the input to this model as output and concatenate it with the input passed through the model.
    def forward(self, x):
        return torch.cat([x, self.model(x)], 1)
