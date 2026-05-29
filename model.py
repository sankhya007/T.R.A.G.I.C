#Unet model it is
# import torch
# import torch.nn as nn

# class DoubleConv(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_c, out_c, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_c, out_c, 3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv(x)


# class UNet(nn.Module):
#     def __init__(self, n_classes=4):
#         super().__init__()

#         self.down1 = DoubleConv(3, 64)
#         self.pool1 = nn.MaxPool2d(2)

#         self.down2 = DoubleConv(64, 128)
#         self.pool2 = nn.MaxPool2d(2)

#         self.down3 = DoubleConv(128, 256)
#         self.pool3 = nn.MaxPool2d(2)

#         self.middle = DoubleConv(256, 512)

#         self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.conv3 = DoubleConv(512, 256)

#         self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.conv2 = DoubleConv(256, 128)

#         self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.conv1 = DoubleConv(128, 64)

#         self.out = nn.Conv2d(32, n_classes, 1)

#     def forward(self, x):
#         d1 = self.down1(x)
#         d2 = self.down2(self.pool1(d1))
#         d3 = self.down3(self.pool2(d2))

#         mid = self.middle(self.pool2(d3))

#         u3 = self.up3(mid)
#         u3 = torch.cat([u3, d3], dim=1)
#         u3 = self.conv3(u3)

#         u2 = self.up2(u3)
#         u2 = torch.cat([u2, d2], dim=1)
#         u2 = self.conv2(u2)

#         u1 = self.up1(u2)
#         u1 = torch.cat([u1, d1], dim=1)
#         u1 = self.conv1(u1)

#         return self.out(u1)








import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.d1 = DoubleConv(3, 64)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = DoubleConv(64, 128)
        self.p2 = nn.MaxPool2d(2)

        self.d3 = DoubleConv(128, 256)
        self.p3 = nn.MaxPool2d(2)

        # Bottleneck
        self.b = DoubleConv(256, 512)

        # Decoder
        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c3 = DoubleConv(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c2 = DoubleConv(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c1 = DoubleConv(128, 64)

        # Output
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))

        b = self.b(self.p3(d3))

        u3 = self.u3(b)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.c3(u3)

        u2 = self.u2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.c2(u2)

        u1 = self.u1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.c1(u1)

        return self.out(u1)