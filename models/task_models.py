import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet3D_regressor(nn.Module):

    def __init__(self, input_size, nf=8):
        super(LeNet3D_regressor, self).__init__()
        num_features = (input_size // (2**2)) **3
        self.conv1 = nn.Conv3d(1, nf, kernel_size=5, padding=2)
        self.conv2 = nn.Conv3d(nf, nf*2, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(nf*2 * num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc22 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.max_pool3d(F.relu(self.conv1(x)), 2)
        x = F.max_pool3d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc22(x))
        x = self.fc3(x).squeeze()

        return x


class LeNet3D_classifier(nn.Module):
    def __init__(self, num_classes):
        super(LeNet3D_classifier, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv0 = nn.Conv3d(1, 8, kernel_size=5, padding=2, stride=2) #64-32
        self.conv1 = nn.Conv3d(8, 16, kernel_size=5, padding=2, stride=2) #32-16
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=5, padding=2, stride=2) #16-8
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=5, padding=2, stride=2) #8-4
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=5, padding=2, stride=2) #4-2
        self.bn4 = nn.BatchNorm3d(128)
        self.conv5 = nn.Conv3d(128, 128, kernel_size=5, padding=2, stride=2) #2-1
        self.bn5 = nn.BatchNorm3d(128)
        self.output = nn.Conv3d(128, num_classes, kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.output(x).squeeze()
        return torch.sigmoid(x)


class Discriminator(nn.Module):
    def __init__(self, nf=32):
        super(Discriminator, self).__init__()

        self.conv0 = nn.Conv3d(1, nf, kernel_size=3, padding=1, stride=1) #64-64
        self.conv1 = nn.Conv3d(nf, nf*2, kernel_size=3, padding=1, stride=2)  # 64-32
        self.bn1 = nn.InstanceNorm3d(nf*2)
        self.conv2 = nn.Conv3d(nf*2, nf*4, kernel_size=3, padding=1, stride=2) #32-16
        self.bn2 = nn.InstanceNorm3d(nf*4)
        self.conv3 = nn.Conv3d(nf*4, nf*8, kernel_size=3, padding=1, stride=2) #16-8
        self.bn3 = nn.InstanceNorm3d(nf*8)
        self.conv4 = nn.Conv3d(nf * 8, nf * 8, kernel_size=3, padding=1, stride=2)  # 8-4
        self.bn4 = nn.InstanceNorm3d(nf * 8)
        self.drop = nn.Dropout(0.5)
        self.output = nn.Conv3d(nf*8, 1, kernel_size=4, padding=0, stride=2) #4-1

    def forward(self, x, return_logits=False, mask=None):
        x = F.relu(self.conv0(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop(x)
        x = self.output(x).squeeze()

        return x if return_logits else torch.sigmoid(x)