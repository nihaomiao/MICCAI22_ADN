# use NN to find symmetric plane, the transformation network T in our paper.
# Some codes are borrowed from https://github.com/elliottwu/unsup3d
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision.transforms as transforms


def get_transform_matrices(view):
    b = view.size(0)
    if view.size(1) == 6:
        rx = view[:, 0]
        ry = view[:, 1]
        rz = view[:, 2]
        trans_xyz = view[:, 3:].reshape(b, 1, 3)
    elif view.size(1) == 5:
        rx = view[:, 0]
        ry = view[:, 1]
        rz = view[:, 2]
        delta_xy = view[:, 3:].reshape(b, 1, 2)
        trans_xyz = torch.cat([delta_xy, torch.zeros(b, 1, 1).to(view.device)], 2)
    elif view.size(1) == 3:
        rx = view[:, 0]
        ry = view[:, 1]
        rz = view[:, 2]
        trans_xyz = torch.zeros(b, 1, 3).to(view.device)
    rot_mat = get_rotation_matrix(rx, ry, rz)
    # change rot_mat to [4, 4]
    R = torch.zeros((rot_mat.shape[0], 4, 4)).to(device=rot_mat.device)
    R[:, :3, :3] = rot_mat
    R[:, 3, 3] = 1
    T = get_translation_matrix(trans_xyz)
    M = torch.matmul(T, R)
    # compute inverse matrix
    R_inv = R.transpose(1, 2)
    T_inv = get_translation_matrix(-trans_xyz)
    M_inv = torch.matmul(R_inv, T_inv)
    return M, M_inv


def get_rotation_matrix(tx, ty, tz):
    m_x = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_y = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_z = torch.zeros((len(tx), 3, 3)).to(tx.device)

    m_x[:, 1, 1], m_x[:, 1, 2] = tx.cos(), -tx.sin()
    m_x[:, 2, 1], m_x[:, 2, 2] = tx.sin(), tx.cos()
    m_x[:, 0, 0] = 1

    m_y[:, 0, 0], m_y[:, 0, 2] = ty.cos(), ty.sin()
    m_y[:, 2, 0], m_y[:, 2, 2] = -ty.sin(), ty.cos()
    m_y[:, 1, 1] = 1

    m_z[:, 0, 0], m_z[:, 0, 1] = tz.cos(), -tz.sin()
    m_z[:, 1, 0], m_z[:, 1, 1] = tz.sin(), tz.cos()
    m_z[:, 2, 2] = 1
    return torch.matmul(m_z, torch.matmul(m_y, m_x))


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


class ConfNet3D(nn.Module):
    # cout = 2, one for original, one for flip
    def __init__(self, cin, cout=2, nf=64, n_downsampling=3):
        super(ConfNet3D, self).__init__()
        network = []
        # downsampling
        for ii in range(n_downsampling):
            c_in = cin if ii == 0 else nf * (2 ** ii) // 2
            c_out = nf * (2 ** ii)
            network += [[nn.Conv3d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.GroupNorm(16 * (2 ** ii), c_out),
                         nn.LeakyReLU(0.2, inplace=True)]]
        # upsampling
        for ii in range(n_downsampling):
            c_in = nf * (2 ** (n_downsampling - 1 - ii))
            c_out = nf * (2 ** (n_downsampling - 1 - ii)) // 2
            network += [[nn.ConvTranspose3d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),  # TODO
                         nn.GroupNorm(16 * (2 ** (n_downsampling - 1 - ii)), c_out),
                         nn.ReLU(inplace=True)]]
        # final layer
        network += [[nn.Conv3d(c_out, cout, kernel_size=5, stride=1, padding=2, bias=False),
                     nn.Softplus()]]
        self.n_layer = len(network)
        for n in range(self.n_layer):
            setattr(self, 'model' + str(n), nn.Sequential(*network[n]))

    def forward(self, x):
        for n in range(self.n_layer):
            model = getattr(self, 'model' + str(n))
            x = model(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x


class Encoder3D(nn.Module):
    def __init__(self, cin=1, cout=6, nf=32, n_downsampling_3d=4):
        super(Encoder3D, self).__init__()
        network3d = []
        # downsampling
        for ii in range(n_downsampling_3d):
            c_in = cin if ii == 0 else nf * (2 ** ii) // 2
            c_out = nf * (2 ** ii)
            network3d += [[nn.Conv3d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),
                           BasicBlock(c_out, c_out, n_groups=8 * (2 ** ii))]]

        self.n_layer_3d = len(network3d)
        for n in range(self.n_layer_3d):
            setattr(self, 'model3d' + str(n), nn.Sequential(*network3d[n]))

        self.avgpool = nn.AvgPool3d((2, 16, 16), stride=1)
        self.fc = nn.Linear(256, cout)

    def forward(self, x):
        for n in range(self.n_layer_3d):
            model = getattr(self, 'model3d' + str(n))
            x = model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.Tanh()(x)
        return x.reshape(x.size(0), -1)


class PlaneFinder(nn.Module):
    def __init__(self, is_train=True):
        super(PlaneFinder, self).__init__()
        self.is_train = is_train
        self.x_rotation_range = 0
        self.y_rotation_range = 0
        self.z_rotation_range = 40
        self.x_translation_range = 0.5
        self.y_translation_range = 0
        self.z_translation_range = 0
        # rotation angle and translation
        self.netV = Encoder3D(cin=1, cout=6, nf=32)
        self.network_names = [k for k in vars(self) if 'net' in k]

    def stn(self, x, theta):
        # theta must be (Bs, 3, 4) = [R|t]
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        out = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        return out

    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None):
        loss = (im1-im2).abs()
        if conf_sigma is not None:
            loss = loss *2**0.5 / (conf_sigma + 1e-7) + (conf_sigma +1e-7).log()
        if mask is not None:
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def forward(self, x):
        # view: x, y, z rotation and translation
        view = self.netV(x)
        # https://medium.com/@daniel.j.lenton/part-iii-projective-geometry-in-3d-37f36746733b
        new_view = torch.cat([view[:, :1] * math.pi / 180 * self.x_rotation_range,
                              view[:, 1:2] * math.pi / 180 * self.y_rotation_range,
                              view[:, 2:3] * math.pi / 180 * self.z_rotation_range,
                              view[:, 3:4] * self.x_translation_range,
                              view[:, 4:5] * self.y_translation_range,
                              view[:, 5:] * self.z_translation_range], 1)
        # M: transformation matrix
        M, M_inv = get_transform_matrices(new_view)
        # transformed volume; should be symmetric
        x_t = self.stn(x, M[:, :3, :])
        # flipped x_t, should be "similar" to x_t due to symmetry
        x_t_f = transforms.functional.hflip(x_t)
        # reconstruct volume; should be "similar" to original x
        x_r_f = self.stn(x_t_f, M_inv[:, :3, :])
        if self.is_train:
            self.loss_flip = self.photometric_loss(x_t, x_t_f)
            self.loss_rec = self.photometric_loss(x, x_r_f)
            self.loss_total = self.loss_flip + self.loss_rec

        return x_t, x_r_f, x_t_f, view, M, M_inv


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = PlaneFinder(is_train=True)
    model.cuda()
    dummy_x = torch.rand((4, 1, 45, 256, 256))
    out = model(dummy_x.cuda())
