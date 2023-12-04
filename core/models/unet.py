
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary




###############################################################
#                   U-Net Assembly Blocks                     #
###############################################################




class ConvBlock(nn.Module):
    """
    conv_block: [Conv -> BN -> ReLU] ** 2
    -------------------------------------
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3) -> None:
        super(ConvBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding='same', bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding='same', bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x:torch.Tensor):
        return self.conv_block(x)


class DownScale(nn.Module):
    """
    Downscaling block equipped with conv_block
    maxpool_conv: [MaxPool -> ConvBlock]
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3) -> None:
        super(DownScale, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            ConvBlock(in_channels, out_channels, mid_channels, kernel_size)
        )

    def forward(self, x:torch.Tensor):
        return self.maxpool_conv(x)


class UpScale(nn.Module):

    def __init__(self, in_channels, out_channels, mode='nearest') -> None:
        """
        mode : ['transpose', 'nearest', 'bilinear']
        """
        super(UpScale, self).__init__()

        if mode == 'transpose':
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)
        else:
            if mode == 'nearest':
                self.up = nn.Upsample(scale_factor=2, mode=mode)
            else:
                self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)


    def forward(self, up_x:torch.Tensor, res_x:torch.Tensor):

        up_x = self.up(up_x) # upscale

        size_diff = torch.tensor(res_x.shape) - torch.tensor(up_x.shape)
        p3d = (torch.div(size_diff[-1], 2, rounding_mode='trunc'), size_diff[-1] - torch.div(size_diff[-1], 2, rounding_mode='trunc'),
               torch.div(size_diff[-2], 2, rounding_mode='trunc'), size_diff[-2] - torch.div(size_diff[-2], 2, rounding_mode='trunc'),
               torch.div(size_diff[-3], 2, rounding_mode='trunc'), size_diff[-3] - torch.div(size_diff[-3], 2, rounding_mode='trunc'))

        up_x = F.pad(up_x, p3d)

        res_concat = torch.concat([res_x, up_x], dim=1) # residual connection

        return self.conv(res_concat)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


###############################################################
#                      U-Net Definition                       #
###############################################################


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, mode='nearest') -> None:
        super(UNet, self).__init__()

        self.inc = ConvBlock(n_channels, 32)
        self.down1 = DownScale(32, 64)
        self.down2 = DownScale(64, 128)
        self.down3 = DownScale(128, 256)
        factor = 2 if not mode == 'transpose' else 1
        self.down4 = DownScale(256, int(512 / factor))
        self.up1 = UpScale(512, int(256 / factor), mode)
        self.up2 = UpScale(256, int(128 / factor), mode)
        self.up3 = UpScale(128, int(64 / factor), mode)
        self.up4 = UpScale(64, 32, mode)
        self.outc = OutConv(32, n_classes)

    def forward(self, x:torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))


if __name__ == "__main__":

    input = torch.ones((1, 64, 64, 64)).to('cuda')

    unet = UNet(1, 3).to('cuda')

    summary(unet, input.shape, batch_size=2, device='cuda')

    print(unet(torch.unsqueeze(input, dim=0)).shape)

