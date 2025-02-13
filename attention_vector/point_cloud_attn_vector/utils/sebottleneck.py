import torch.nn as nn

class SEBottleneck(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        squeeze = self.squeeze(x).view(b, c)
        excite = self.excite(squeeze).view(b, c, 1)
        return x * excite
