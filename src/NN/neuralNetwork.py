import torch
import torch.nn as nn

class FluidPDEAI(nn.Module):
    """
    A Convolutional Neural Network to predict the next fluid simulation step.
    Takes in multiple channels (u, v, p, obstacle, dt, viscosity) and predicts u, v, p at t+dt.
    """
    def __init__(self, in_channels=6, out_channels=3):
        """
        Args:
            in_channels (int): Number of input channels (default 6)
            out_channels (int): Number of output channels (default 3)
        """
        super(FluidPDEAI, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)  # No padding needed
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x
