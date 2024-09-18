import torch
import torch.nn.functional as F
from einops import reduce
import torch.nn as nn
device = torch.device("cuda")

class BasicUNet(nn.Module):
    """A minimal UNet implementation."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels + 1, 128, kernel_size=3, padding=1, groups=1),   
            nn.Conv2d(128, 512, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=16),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(512, 128, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1, groups=1),
        ])
        self.upsample_layers = torch.nn.ModuleList([
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
        ])
        # self.upscale = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        self.act = nn.SiLU() # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)


    def forward(self, x, t):
        # concatenate the timestep to the input
        t = torch.ones_like(x[0]) * t[:,None,None,None] / 1000.0
        t = reduce(t, 'b c h w -> b 1 h w', 'mean')
        x = torch.cat([x, t], dim=1)
        # make sure correct device
        x = x.to(device)
        
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x)) # Through the layer and the activation function
            if i < (len(self.down_layers)-1): # For all but the third (final) down layer:
              h.append(x) # Storing output for skip connection
              x = self.downscale(x) # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0: # For all except the first up layer
              # x = self.upscale(x) # Upscale
              x = self.upsample_layers[i-1](x)
              x += h.pop() # Fetching stored output (skip connection)
            x = self.act(l(x)) # Through the layer and the activation function

        return x