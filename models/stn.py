import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineSTN3D(nn.Module):

    def __init__(self, input_size, device, input_channels=1, nf=2):
        super(AffineSTN3D, self).__init__()
        self.dtype = torch.float
        self.device = device
        self.nf = nf
        self.conv00 = nn.Conv3d(input_channels, self.nf, kernel_size=3, padding=1).to(self.device)
        self.bn00 = nn.BatchNorm3d(self.nf).to(self.device)
        self.conv0 = nn.Conv3d(self.nf, self.nf * 2, kernel_size=3, padding=1, stride=2).to(self.device)
        self.bn0 = nn.BatchNorm3d(self.nf * 2).to(self.device)
        self.conv1 = nn.Conv3d(self.nf * 2, self.nf * 4, kernel_size=3, padding=1, stride=2).to(self.device)
        self.bn1 = nn.BatchNorm3d(self.nf * 4).to(self.device)
        self.conv2 = nn.Conv3d(self.nf * 4, self.nf * 4, kernel_size=3, padding=1, stride=2).to(self.device)
        self.bn2 = nn.BatchNorm3d(self.nf * 4).to(self.device)
        self.conv3 = nn.Conv3d(self.nf * 4, self.nf * 4, kernel_size=3, padding=1, stride=2).to(self.device)
        self.bn3 = nn.BatchNorm3d(self.nf * 4).to(self.device)
        
        final_size = 4 * 4 * 4* self.nf * 4

        # Regressor for individual parameters
        self.translation = nn.Linear(final_size, 3).to(self.device)
        self.rotation = nn.Linear(final_size, 3).to(self.device)
        self.scaling = nn.Linear(final_size, 3).to(self.device)
        self.shearing = nn.Linear(final_size, 3).to(self.device)

        # initialize the weights/bias with identity transformation
        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.rotation.weight.data.zero_()
        self.rotation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.scaling.weight.data.zero_()
        self.scaling.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.shearing.weight.data.zero_()
        self.shearing.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))

    def get_theta(self, i):
        return self.theta[i]

    def forward(self, x):
        xs = F.relu(self.bn00(self.conv00(x)))
        xs = F.relu(self.bn0(self.conv0(xs)))
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))
        xs = xs.view(xs.size(0), -1)

        self.affine_matrix(xs)

        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img).to(self.device)

    def gen_3d_mesh_grid(self, d, h, w):
        d_s = torch.linspace(-1, 1, d)
        h_s = torch.linspace(-1, 1, h)
        w_s = torch.linspace(-1, 1, w)

        d_s, h_s, w_s = torch.meshgrid([d_s, h_s, w_s])
        one_s = torch.ones_like(w_s)

        mesh_grid = torch.stack([w_s, h_s, d_s, one_s])
        return mesh_grid  # 4 x d x h x w

    def affine_grid(self, theta, size):
        b, c, d, h, w = size
        mesh_grid = self.gen_3d_mesh_grid(d, h, w)
        mesh_grid = mesh_grid.unsqueeze(0)

        mesh_grid = mesh_grid.repeat(b, 1, 1, 1, 1)  # channel dim = 4
        mesh_grid = mesh_grid.view(b, 4, -1)
        mesh_grid = torch.bmm(theta, mesh_grid)  # channel dim = 3
        mesh_grid = mesh_grid.permute(0, 2, 1)  # move channel to last dim
        return mesh_grid.view(b, d, h, w, 3)

    def warp_image(self, img):
        grid = self.affine_grid(self.theta[:, 0:3, :], img.size()).to(self.device)
        wrp = F.grid_sample(img, grid, align_corners=False)

        return wrp

    def warp_inv_image(self, img):
        grid = self.affine_grid(self.theta_inv[:, 0:3, :], img.size()).to(self.device)
        wrp = F.grid_sample(img, grid, align_corners=False)

        return wrp

    def affine_matrix(self, x):
        b = x.size(0)

        ### TRANSLATION ###
        trans = torch.tanh(self.translation(x)) * 0.1
        translation_matrix = torch.zeros([b, 4, 4], dtype=torch.float)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 2, 2] = 1.0
        translation_matrix[:, 0, 3] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 3] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 3] = trans[:, 2].view(-1)
        translation_matrix[:, 3, 3] = 1.0

        ### ROTATION ###
        rot = torch.tanh(self.rotation(x)) * (np.pi / 4.0)
        angle_1 = rot[:, 0].view(-1)
        rotation_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float)
        rotation_matrix_1[:, 0, 0] = torch.cos(angle_1)
        rotation_matrix_1[:, 0, 1] = -torch.sin(angle_1)
        rotation_matrix_1[:, 1, 0] = torch.sin(angle_1)
        rotation_matrix_1[:, 1, 1] = torch.cos(angle_1)
        rotation_matrix_1[:, 2, 2] = 1.0
        rotation_matrix_1[:, 3, 3] = 1.0
        
        angle_2 = rot[:, 1].view(-1)
        rotation_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float)
        rotation_matrix_2[:, 1, 1] = torch.cos(angle_2)
        rotation_matrix_2[:, 1, 2] = -torch.sin(angle_2)
        rotation_matrix_2[:, 2, 1] = torch.sin(angle_2)
        rotation_matrix_2[:, 2, 2] = torch.cos(angle_2)
        rotation_matrix_2[:, 0, 0] = 1.0
        rotation_matrix_2[:, 3, 3] = 1.0
        
        angle_3 = rot[:, 2].view(-1)
        rotation_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float)
        rotation_matrix_3[:, 0, 0] = torch.cos(angle_3)
        rotation_matrix_3[:, 0, 1] = -torch.sin(angle_3)
        rotation_matrix_3[:, 1, 0] = torch.sin(angle_3)
        rotation_matrix_3[:, 1, 1] = torch.cos(angle_3)
        rotation_matrix_3[:, 2, 2] = 1.0
        rotation_matrix_3[:, 3, 3] = 1.0

        rotation_matrix = torch.bmm(rotation_matrix_1, rotation_matrix_2)
        rotation_matrix = torch.bmm(rotation_matrix, rotation_matrix_3)

        ### SCALING ###
        scale = torch.tanh(self.scaling(x)) * 0.2
        scaling_matrix = torch.zeros([b, 4, 4], dtype=torch.float)
        scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        scaling_matrix[:, 2, 2] = torch.exp(scale[:, 2].view(-1))
        scaling_matrix[:, 3, 3] = 1.0

        ### SHEARING ###
        shear = torch.tanh(self.shearing(x)) * (np.pi / 4.0)

        shear_1 = shear[:, 0].view(-1)
        shearing_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float)
        shearing_matrix_1[:, 1, 1] = torch.cos(shear_1)
        shearing_matrix_1[:, 1, 2] = -torch.sin(shear_1)
        shearing_matrix_1[:, 2, 1] = torch.sin(shear_1)
        shearing_matrix_1[:, 2, 2] = torch.cos(shear_1)
        shearing_matrix_1[:, 0, 0] = 1.0
        shearing_matrix_1[:, 3, 3] = 1.0

        shear_2 = shear[:, 1].view(-1)
        shearing_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float)
        shearing_matrix_2[:, 0, 0] = torch.cos(shear_2)
        shearing_matrix_2[:, 0, 2] = torch.sin(shear_2)
        shearing_matrix_2[:, 2, 0] = -torch.sin(shear_2)
        shearing_matrix_2[:, 2, 2] = torch.cos(shear_2)
        shearing_matrix_2[:, 1, 1] = 1.0
        shearing_matrix_2[:, 3, 3] = 1.0

        shear_3 = shear[:, 2].view(-1)
        shearing_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float)
        shearing_matrix_3[:, 0, 0] = torch.cos(shear_3)
        shearing_matrix_3[:, 0, 1] = -torch.sin(shear_3)
        shearing_matrix_3[:, 1, 0] = torch.sin(shear_3)
        shearing_matrix_3[:, 1, 1] = torch.cos(shear_3)
        shearing_matrix_3[:, 2, 2] = 1.0
        shearing_matrix_3[:, 3, 3] = 1.0

        shearing_matrix = torch.bmm(shearing_matrix_1, shearing_matrix_2)
        shearing_matrix = torch.bmm(shearing_matrix, shearing_matrix_3)

        # Affine transform
        matrix = torch.bmm(shearing_matrix, scaling_matrix)
        matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        matrix = torch.bmm(matrix, rotation_matrix)
        matrix = torch.bmm(matrix, translation_matrix)

        self.theta = matrix
        self.theta_inv = torch.inverse(matrix)
        
        
class BSplineSTN3D(nn.Module):
    """
    B-spline implementation inspired by https://github.com/airlab-unibas/airlab
    """
    def __init__(self, input_size, device, input_channels=1, nf=16, control_point_spacing=(10, 10, 10), bspline_order=3, max_displacement=0.2):
        super(BSplineSTN3D, self).__init__()
        # Cuda params
        self.device = device
        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float

        self.dtype = torch.float

        self.order = bspline_order
        self.max_disp = max_displacement

        self.input_size = input_size
        self.control_point_spacing = np.array(control_point_spacing)
        self.stride = self.control_point_spacing.astype(dtype=int).tolist()

        area = self.control_point_spacing[0] * self.control_point_spacing[1] * self.control_point_spacing[2]
        self.area = area.astype(float)
        cp_grid_shape = np.ceil(np.divide(self.input_size, self.control_point_spacing)).astype(dtype=int)

        # new image size after convolution
        self.inner_image_size = np.multiply(self.control_point_spacing, cp_grid_shape) - (
                self.control_point_spacing - 1)

        # add one control point at each side
        cp_grid_shape = cp_grid_shape + 2

        # image size with additional control points
        self.new_image_size = np.multiply(self.control_point_spacing, cp_grid_shape) - (self.control_point_spacing - 1)

        # center image between control points
        image_size_diff = self.inner_image_size - self.input_size
        image_size_diff_floor = np.floor((np.abs(image_size_diff) / 2)) * np.sign(image_size_diff)
        crop_start = image_size_diff_floor + np.remainder(image_size_diff, 2) * np.sign(image_size_diff)
        self.crop_start = crop_start.astype(dtype=int)
        self.crop_end = image_size_diff_floor.astype(dtype=int)

        self.cp_grid_shape = [3] + cp_grid_shape.tolist()

        self.num_control_points = np.prod(self.cp_grid_shape)
        self.kernel = self.bspline_kernel_3d(order=self.order).expand(3, *((np.ones(3 + 1, dtype=int) * -1).tolist()))
        self.kernel_size = np.asarray(self.kernel.size())[2:]
        self.padding = ((self.kernel_size - 1) / 2).astype(dtype=int).tolist()

        # Network params
        num_features = torch.prod(((((torch.tensor(input_size) - 4) / 2 - 4) / 2) - 4) / 2)
        self.nf = nf
        self.conv00 = nn.Conv3d(input_channels, self.nf, kernel_size=5, padding=2)
        self.bn00 = nn.BatchNorm3d(self.nf)
        self.conv0 = nn.Conv3d(self.nf, self.nf * 2, kernel_size=5, padding=2, stride=2)
        self.bn0 = nn.BatchNorm3d(self.nf * 2)
        self.conv1 = nn.Conv3d(self.nf * 2, self.nf * 4, kernel_size=5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm3d(self.nf * 4)
        self.conv2 = nn.Conv3d(self.nf * 4, self.nf * 8, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm3d(self.nf * 8)
        self.conv3 = nn.Conv3d(self.nf * 8, self.nf * 8, kernel_size=5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm3d(self.nf * 8)
        final_size = (4 **3) * self.nf * 8

        self.fc = nn.Linear(final_size, self.num_control_points)

    def gen_3d_mesh_grid(self, d, h, w):
        d_s = torch.linspace(-1, 1, d)
        h_s = torch.linspace(-1, 1, h)
        w_s = torch.linspace(-1, 1, w)

        d_s, h_s, w_s = torch.meshgrid([d_s, h_s, w_s])

        mesh_grid = torch.stack([w_s, h_s, d_s])
        return mesh_grid.permute(1, 2, 3, 0).to(self.device)  # d x h x w x 3

    def bspline_kernel_3d(self, order):
        kernel_ones = torch.ones(1, 1, *self.control_point_spacing)
        kernel = kernel_ones

        for i in range(1, order + 1):
            kernel = F.conv3d(kernel, kernel_ones, padding=self.control_point_spacing.tolist()) / self.area

        return kernel.to(dtype=self.dtype, device=self.device)

    def compute_displacement(self, params):
        # compute dense displacement
        displacement = F.conv_transpose3d(params, self.kernel,
                                          padding=self.padding, stride=self.stride, groups=3)

        # crop displacement
        displacement = displacement[:, :,
                       self.control_point_spacing[0] + self.crop_start[0]:-self.control_point_spacing[0] -
                                                                          self.crop_end[0],
                       self.control_point_spacing[1] + self.crop_start[1]:-self.control_point_spacing[1] -
                                                                          self.crop_end[1],
                       self.control_point_spacing[2] + self.crop_start[2]:-self.control_point_spacing[2] -
                                                                          self.crop_end[2]]

        return displacement.permute(0, 2, 3, 4, 1)

    def get_theta(self, i):
        return self.control_points[i]

    def forward(self, x):
        b, c, d, h, w = x.shape
        xs = F.relu(self.bn00(self.conv00(x)))
        xs = F.relu(self.bn0(self.conv0(xs)))
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))
        xs = xs.view(xs.size(0), -1)
        self.regularisation_loss = 30.0 * torch.mean(torch.abs(xs))
        # cap the displacement field by (-1,1) this still allows for non-diffeomorphic transformations
        xs = torch.tanh(self.fc(xs)) * self.max_disp
        xs = xs.view(-1, *self.cp_grid_shape)


        self.displacement_field = self.compute_displacement(xs) + self.gen_3d_mesh_grid(d, h, w).unsqueeze(0)
        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img)

    def warp_image(self, img):
        wrp = F.grid_sample(img, self.displacement_field, mode='bilinear', align_corners=False)

        return wrp