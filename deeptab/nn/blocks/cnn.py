import torch.nn as nn


class CNNBlock(nn.Module):
    """A modular CNN block that allows for configurable convolutional, pooling, and dropout layers.

    Attributes
    ----------
    cnn : nn.Sequential
        A sequential container holding the convolutional, activation, pooling, and dropout layers.

    Methods
    -------
    forward(x):
        Defines the forward pass of the CNNBlock.
    """

    def __init__(self, config):
        super().__init__()
        layers = []
        in_channels = config.input_channels

        # Ensure dropout_positions is a list
        dropout_positions = config.dropout_positions or []

        for i in range(config.num_layers):
            # Convolutional layer
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=config.out_channels_list[i],
                    kernel_size=config.kernel_size_list[i],
                    stride=config.stride_list[i],
                    padding=config.padding_list[i],
                )
            )
            layers.append(nn.ReLU())

            # Pooling layer
            if config.pooling_method == "max":
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=config.pooling_kernel_size_list[i],
                        stride=config.pooling_stride_list[i],
                    )
                )
            elif config.pooling_method == "avg":
                layers.append(
                    nn.AvgPool2d(
                        kernel_size=config.pooling_kernel_size_list[i],
                        stride=config.pooling_stride_list[i],
                    )
                )

            # Dropout layer
            if i in dropout_positions:
                layers.append(nn.Dropout(p=config.dropout_rate))

            in_channels = config.out_channels_list[i]

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        # Ensure input has shape (N, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.cnn(x)
