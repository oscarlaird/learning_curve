import torch
import torch.nn as nn

class BasicUNet(nn.Module):
    """A minimal UNet implementation."""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, groups=1),   
            nn.Conv2d(256, 512, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, groups=16),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(256, out_channels, kernel_size=3, padding=1, groups=1),
        ])

        self.act = nn.ReLU() # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)


    def forward(self, x, t):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x)) # Through the layer and the activation function
            if i < (len(self.down_layers)-1): # For all but the third (final) down layer:
              h.append(x) # Storing output for skip connection
              x = self.downscale(x) # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0: # For all except the first up layer
              x = self.upscale(x) # Upscale
              x += h.pop() # Fetching stored output (skip connection)
            x = l(x)
            if i < (len(self.up_layers)-1):
                x = self.act(x) # Activation function for all but the final layer

        return x