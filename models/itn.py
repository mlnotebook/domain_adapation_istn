import torch
import torch.nn as nn
import torch.nn.functional as F

class ITN3D(nn.Module):

    def __init__(self, nf=16):
        super(ITN3D, self).__init__()

        self.conv0 = nn.Conv3d(1, nf, kernel_size=3, padding=1) #64-64
        self.bn0 = nn.BatchNorm3d(nf)
        self.conv1 = nn.Conv3d(nf, nf*2, kernel_size=3, padding=1, stride=2) #64-32
        self.bn1 = nn.BatchNorm3d(nf*2)
        self.conv2 = nn.Conv3d(nf*2, nf*4, kernel_size=3, padding=1, stride=2) #32-16
        self.bn2 = nn.BatchNorm3d(nf*4)
        self.conv3 = nn.Conv3d(nf * 4, nf * 8, kernel_size=3, padding=1, stride=2)  # 16-8
        self.bn3 = nn.BatchNorm3d(nf * 8)

        self.bottleneck0 = nn.Conv3d(nf*8, nf*8, kernel_size=3, padding=1) #8-8
        self.bnb0 = nn.BatchNorm3d(nf * 8)
        self.bottleneck1 = nn.Conv3d(nf*8, nf*8, kernel_size=3, padding=1) #8-8
        self.bnb1 = nn.BatchNorm3d(nf * 8)

        self.up31 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # 8-16
        self.pad3 = nn.ConstantPad3d(1, 0)
        self.up32 = nn.Conv3d(nf * 8, nf * 4, kernel_size=3, padding=0)
        self.drop3 = nn.Dropout(0.5)

        self.up21 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) #16-32
        self.pad2 = nn.ConstantPad3d(1, 0)
        self.up22 = nn.Conv3d(nf*4 + nf*4, nf*2, kernel_size=3, padding=0)
        self.drop2 = nn.Dropout(0.5)

        self.up11 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) #32-64
        self.pad1 = nn.ConstantPad3d(1, 0)
        self.up12 = nn.Conv3d(nf*2 + nf*2, nf, kernel_size=3, padding=0)
        self.drop1 = nn.Dropout(0.5)

        self.pad0 = nn.ConstantPad3d(1, 0)
        self.output = nn.Conv3d(nf + nf, 1, kernel_size=3, padding=0)

    def forward(self, x):
        c0 = F.relu(self.bn0(self.conv0(x)))
        c1 = F.relu(self.bn1(self.conv1(c0)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c3 = F.relu(self.bn3(self.conv3(c2)))

        b0 = F.relu(self.bnb0(self.bottleneck0(c3)))
        b1 = F.relu(self.bnb1(self.bottleneck1(b0)))

        u3 = F.relu(self.up32(self.pad3(self.up31(b1))))
        u3cat = self.drop3(torch.cat([u3, c2], 1))
        u2 = F.relu(self.up22(self.pad2(self.up21(u3cat))))
        u2cat = self.drop2(torch.cat([u2, c1], 1))
        u1 = F.relu(self.up12(self.pad1(self.up11(u2cat))))
        u1cat = self.drop1(torch.cat([u1, c0], 1))
        out = self.output(self.pad0(u1cat)) + x

        return torch.tanh(out)