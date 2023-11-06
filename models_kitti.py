import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
import utils
import os
import torchvision.transforms.functional as TF

# from GRU1 import ElevationEsitimate,VisibilityEsitimate,VisibilityEsitimate2,GRUFuse
from VGG import VGGUnet, VGGUnet_G2S, Encoder, Decoder, Decoder2, Decoder4, VGGUnetTwoDec
from jacobian import grid_sample

# from ConvLSTM import VE_LSTM3D, VE_LSTM2D, VE_conv, S_LSTM2D
# from models_ford import loss_func
from RNNs import NNrefine, Uncertainty, VisibilityMask
from swin_transformer import TransOptimizerS2GP_V1, TransOptimizerG2SP_V1, TransOptimizerG2SP_V2
from swin_transformer_cross import TransOptimizerG2SP, TransOptimizerG2SPV2, SwinTransformerSelf
from cross_attention import CrossViewAttention
import copy
from extracter import BasicEncoder, SmallEncoder
import cv2

import matplotlib.pyplot as plt
import cv2
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors

EPS = utils.EPS


class Model(nn.Module):
    def __init__(self, args, device=None):  # device='cuda:0',
        super(Model, self).__init__()

        self.args = args
        self.device = device

        self.level = sorted([int(item) for item in args.level.split('_')])
        self.N_iters = args.N_iters
        self.channels = [int(item) for item in self.args.channels.split('_')]

        self.SatFeatureNet = VGGUnet(self.level, self.channels)

        if self.args.proj == 'CrossAttn':
            self.Dec4 = Decoder4(self.channels[0])
            self.Dec2 = Decoder2(self.channels[0:2])
            self.CVattn = CrossViewAttention(blocks=2, dim=256, heads=4, dim_head=16, qkv_bias=False)

        if self.args.share:
            self.FeatureForT = VGGUnet(self.level, self.channels)
        else:
            self.GrdFeatureForT = VGGUnet(self.level, self.channels)
            self.SatFeatureForT = VGGUnet(self.level, self.channels)

        self.meters_per_pixel = {}
        meter_per_pixel = utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel[level] = meter_per_pixel * (2 ** (3 - level))

        self.TransRefine = TransOptimizerG2SP_V1(self.channels)

        if self.args.ConfGrd == 2:
            self.VisMask = VisibilityMask(dims=[256] + self.channels)

        self.coe_R = nn.Parameter(torch.tensor(-5., dtype=torch.float32), requires_grad=True)
        self.coe_T = nn.Parameter(torch.tensor(-3., dtype=torch.float32), requires_grad=True)

        # if self.args.use_uncertainty:
        #     self.uncertain_net = Uncertainty(self.channels)

        self.masks = {}
        for level in range(4):
            A = 512 / 2**(3-level)
            XYZ_1 = self.sat2world(A)  # [ sidelength,sidelength,4]

            B = 1
            shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
            shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
            heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)

            ori_camera_k = torch.tensor([[[582.9802, 0.0000, 496.2420],
                                          [0.0000, 482.7076, 125.0034],
                                          [0.0000, 0.0000, 1.0000]]],
                                        dtype=torch.float32, requires_grad=True, device=self.device)
            ori_grdH, ori_grdW = 256, 1024
            H, W = ori_grdH, ori_grdW

            uv, mask = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, ori_camera_k, H, W,
                                                       ori_grdH, ori_grdW)
            # [B, H, W, 2], [B, H, W, 1]
            self.masks[level] = mask[:, :, :, 0]

        torch.autograd.set_detect_anomaly(True)
        # Running the forward pass with detection enabled will allow the backward pass to print the traceback of the forward operation that created the failing backward function.
        # Any backward computation that generate “nan” value will raise an error.

    def initialize_grd_encoder(self):
        self.GrdFeatureNet = copy.deepcopy(self.SatFeatureNet)

    def initialize_grd_encoder_for_T(self):
        self.GrdFeatureForT = copy.deepcopy(self.SatFeatureNet)

    def grd_img2cam(self, grd_H, grd_W, ori_grdH, ori_grdW):

        ori_camera_k = torch.tensor([[[582.9802, 0.0000, 496.2420],
                                      [0.0000, 482.7076, 125.0034],
                                      [0.0000, 0.0000, 1.0000]]],
                                    dtype=torch.float32, requires_grad=True)  # [1, 3, 3]

        camera_height = utils.get_camera_height()

        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1,
                             :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        camera_k_inv = torch.inverse(camera_k)  # [B, 3, 3]

        v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32),
                              torch.arange(0, grd_W, dtype=torch.float32))
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).unsqueeze(dim=0)  # [1, grd_H, grd_W, 3]
        xyz_w = torch.sum(camera_k_inv[:, None, None, :, :] * uv1[:, :, :, None, :], dim=-1)  # [1, grd_H, grd_W, 3]

        w = camera_height / torch.where(torch.abs(xyz_w[..., 1:2]) > utils.EPS, xyz_w[..., 1:2],
                                        utils.EPS * torch.ones_like(xyz_w[..., 1:2]))  # [BN, grd_H, grd_W, 1]
        xyz_grd = xyz_w * w  # [1, grd_H, grd_W, 3] under the grd camera coordinates
        # xyz_grd = xyz_grd.reshape(B, N, grd_H, grd_W, 3)

        mask = (xyz_grd[..., -1] > 0).float()  # # [1, grd_H, grd_W]

        return xyz_grd, mask, xyz_w

    def grd2cam2world2sat(self, ori_shift_u, ori_shift_v, ori_heading, level,
                          satmap_sidelength, require_jac=False, gt_depth=None):
        '''
        realword: X: south, Y:down, Z: east
        camera: u:south, v: down from center (when heading east, need to rotate heading angle)
        Args:
            ori_shift_u: [B, 1]
            ori_shift_v: [B, 1]
            heading: [B, 1]
            XYZ_1: [H,W,4]
            ori_camera_k: [B,3,3]
            grd_H:
            grd_W:
            ori_grdH:
            ori_grdW:

        Returns:
        '''
        B, _ = ori_heading.shape
        heading = ori_heading * self.args.rotation_range / 180 * np.pi
        shift_u = ori_shift_u * self.args.shift_range_lon
        shift_v = ori_shift_v * self.args.shift_range_lat

        cos = torch.cos(heading)
        sin = torch.sin(heading)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B, 9]
        R = R.view(B, 3, 3)  # shape = [B, N, 3, 3]
        # this R is the inverse of the R in G2SP

        camera_height = utils.get_camera_height()
        # camera offset, shift[0]:east,Z, shift[1]:north,X
        height = camera_height * torch.ones_like(shift_u[:, :1])
        T0 = torch.cat([shift_v, height, -shift_u], dim=-1)  # shape = [B, 3]
        # T0 = torch.unsqueeze(T0, dim=-1)  # shape = [B, N, 3, 1]
        # T = torch.einsum('bnij, bnj -> bni', -R, T0) # [B, N, 3]
        T = torch.sum(-R * T0[:, None, :], dim=-1)  # [B, 3]

        # The above R, T define transformation from camera to world

        if self.args.use_gt_depth and gt_depth != None:
            xyz_w = self.xyz_grds[level][2].detach().to(ori_shift_u.device).repeat(B, 1, 1, 1)
            H, W = xyz_w.shape[1:-1]
            depth = F.interpolate(gt_depth[:, None, :, :], (H, W))
            xyz_grd = xyz_w * depth.permute(0, 2, 3, 1)
            mask = (gt_depth != -1).float()
            mask = F.interpolate(mask[:, None, :, :], (H, W), mode='nearest')
            mask = mask[:, 0, :, :]
        else:
            xyz_grd = self.xyz_grds[level][0].detach().to(ori_shift_u.device).repeat(B, 1, 1, 1)
            mask = self.xyz_grds[level][1].detach().to(ori_shift_u.device).repeat(B, 1, 1)  # [B, grd_H, grd_W]
        grd_H, grd_W = xyz_grd.shape[1:3]

        xyz = torch.sum(R[:, None, None, :, :] * xyz_grd[:, :, :, None, :], dim=-1) + T[:, None, None, :]
        # [B, grd_H, grd_W, 3]
        # zx0 = torch.stack([xyz[..., 2], xyz[..., 0]], dim=-1)  # [B, N, grd_H, grd_W, 2]
        R_sat = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.float32, device=ori_shift_u.device, requires_grad=True) \
            .reshape(2, 3)
        zx = torch.sum(R_sat[None, None, None, :, :] * xyz[:, :, :, None, :], dim=-1)
        # [B, grd_H, grd_W, 2]
        # assert zx == zx0

        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength
        sat_uv = zx / meter_per_pixel + satmap_sidelength / 2  # [B, grd_H, grd_W, 2] sat map uv

        if require_jac:
            dR_dtheta = self.args.rotation_range / 180 * np.pi * \
                        torch.cat([-sin, zeros, -cos, zeros, zeros, zeros, cos, zeros, -sin],
                                  dim=-1)  # shape = [B, N, 9]
            dR_dtheta = dR_dtheta.view(B, 3, 3)
            # R_zeros = torch.zeros_like(dR_dtheta)

            dT0_dshiftu = self.args.shift_range_lon * torch.tensor([0., 0., -1.], dtype=torch.float32,
                                                                   device=shift_u.device,
                                                                   requires_grad=True).view(1, 3).repeat(B, 1)
            dT0_dshiftv = self.args.shift_range_lat * torch.tensor([1., 0., 0.], dtype=torch.float32,
                                                                   device=shift_u.device,
                                                                   requires_grad=True).view(1, 3).repeat(B, 1)
            # T0_zeros = torch.zeros_like(dT0_dx)

            dxyz_dshiftu = torch.sum(-R * dT0_dshiftu[:, None, :], dim=-1)[:, None, None, :]. \
                repeat([1, grd_H, grd_W, 1])  # [B, grd_H, grd_W, 3]
            dxyz_dshiftv = torch.sum(-R * dT0_dshiftv[:, None, :], dim=-1)[:, None, None, :]. \
                repeat([1, grd_H, grd_W, 1])  # [B, grd_H, grd_W, 3]
            dxyz_dtheta = torch.sum(dR_dtheta[:, None, None, :, :] * xyz_grd[:, :, :, None, :], dim=-1) + \
                          torch.sum(-dR_dtheta * T0[:, None, :], dim=-1)[:, None, None, :]

            duv_dshiftu = 1 / meter_per_pixel * \
                          torch.sum(R_sat[None, None, None, :, :] * dxyz_dshiftu[:, :, :, None, :], dim=-1)
            # [B, grd_H, grd_W, 2]
            duv_dshiftv = 1 / meter_per_pixel * \
                          torch.sum(R_sat[None, None, None, :, :] * dxyz_dshiftv[:, :, :, None, :], dim=-1)
            # [B, grd_H, grd_W, 2]
            duv_dtheta = 1 / meter_per_pixel * \
                         torch.sum(R_sat[None, None, None, :, :] * dxyz_dtheta[:, :, :, None, :], dim=-1)
            # [B, grd_H, grd_W, 2]

            # duv_dshift = torch.stack([duv_dx, duv_dy], dim=0)
            # duv_dtheta = duv_dtheta.unsqueeze(dim=0)

            return sat_uv, mask, duv_dshiftu, duv_dshiftv, duv_dtheta

        return sat_uv, mask, None, None, None

    def project_map_to_grd(self, sat_f, sat_c, shift_u, shift_v, heading, level, require_jac=True, gt_depth=None):
        '''
        Args:
            sat_f: [B, C, H, W]
            sat_c: [B, 1, H, W]
            shift_u: [B, 2]
            shift_v: [B, 2]
            heading: [B, 1]
            camera_k: [B, 3, 3]

            ori_grdH:
            ori_grdW:

        Returns:

        '''
        B, C, satmap_sidelength, _ = sat_f.size()
        A = satmap_sidelength

        uv, mask, jac_shiftu, jac_shiftv, jac_heading = self.grd2cam2world2sat(shift_u, shift_v, heading, level,
                                                                               satmap_sidelength, require_jac, gt_depth)
        # [B, H, W, 2], [B, H, W], [B, H, W, 2], [B, H, W, 2], [B,H, W, 2]

        B, grd_H, grd_W, _ = uv.shape
        if require_jac:
            jac = torch.stack([jac_shiftu, jac_shiftv, jac_heading], dim=0)  # [3, B, H, W, 2]

            # jac = jac.reshape(3, -1, grd_H, grd_W, 2)
        else:
            jac = None

        # print('==================')
        # print(jac.shape)
        # print('==================')

        sat_f_trans, new_jac = grid_sample(sat_f,
                                           uv,
                                           jac)
        sat_f_trans = sat_f_trans * mask[:, None, :, :]
        if require_jac:
            new_jac = new_jac * mask[None, :, None, :, :]

        if sat_c is not None:
            sat_c_trans, _ = grid_sample(sat_c, uv)
            sat_c_trans = sat_c_trans * mask[:, None, :, :]
        else:
            sat_c_trans = None

        return sat_f_trans, sat_c_trans, new_jac, uv * mask[:, :, :, None], mask

    def sat2world(self, satmap_sidelength):
        # satellite: u:east , v:south from bottomleft and u_center: east; v_center: north from center
        # realword: X: south, Y:down, Z: east   origin is set to the ground plane

        # meshgrid the sat pannel
        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        ii, jj = torch.meshgrid(i, j)  # i:h,j:w

        # uv is coordinate from top/left, v: south, u:east
        uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_sidelength, satmap_sidelength, 2]

        # sat map from top/left to center coordinate
        u0 = v0 = satmap_sidelength // 2
        uv_center = uv - torch.tensor(
            [u0, v0]).cuda()  # .to(self.device) # shape = [satmap_sidelength, satmap_sidelength, 2]

        # affine matrix: scale*R
        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength
        R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        # Trans matrix from sat to realword
        XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                          uv_center)  # shape = [satmap_sidelength, satmap_sidelength, 2]

        Y = torch.zeros_like(XZ[..., 0:1])
        ones = torch.ones_like(Y)
        sat2realwap = torch.cat([XZ[:, :, :1], Y, XZ[:, :, 1:], ones], dim=-1)  # [sidelength,sidelength,4]

        return sat2realwap

    def World2GrdImgPixCoordinates(self, ori_shift_u, ori_shift_v, ori_heading, XYZ_1, ori_camera_k, grd_H, grd_W, ori_grdH,
                             ori_grdW):
        # realword: X: south, Y:down, Z: east
        # camera: u:south, v: down from center (when heading east, need to rotate heading angle)
        # XYZ_1:[H,W,4], heading:[B,1], camera_k:[B,3,3], shift:[B,2]
        B = ori_heading.shape[0]
        shift_u_meters = self.args.shift_range_lon * ori_shift_u
        shift_v_meters = self.args.shift_range_lat * ori_shift_v
        heading = ori_heading * self.args.rotation_range / 180 * np.pi

        cos = torch.cos(-heading)
        sin = torch.sin(-heading)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B,9]
        R = R.view(B, 3, 3)  # shape = [B,3,3]

        camera_height = utils.get_camera_height()
        # camera offset, shift[0]:east,Z, shift[1]:north,X
        height = camera_height * torch.ones_like(shift_u_meters)
        T = torch.cat([shift_v_meters, height, -shift_u_meters], dim=-1)  # shape = [B, 3]
        T = torch.unsqueeze(T, dim=-1)  # shape = [B,3,1]
        # T = torch.einsum('bij, bjk -> bik', R, T0)
        # T = R @ T0

        # P = K[R|T]
        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1,
                             :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        # P = torch.einsum('bij, bjk -> bik', camera_k, torch.cat([R, T], dim=-1)).float()  # shape = [B,3,4]
        P = camera_k @ torch.cat([R, T], dim=-1)

        # uv1 = torch.einsum('bij, hwj -> bhwi', P, XYZ_1)  # shape = [B, H, W, 3]
        uv1 = torch.sum(P[:, None, None, :, :] * XYZ_1[None, :, :, None, :], dim=-1)
        # only need view in front of camera ,Epsilon = 1e-6
        uv1_last = torch.maximum(uv1[:, :, :, 2:], torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
        uv = uv1[:, :, :, :2] / uv1_last  # shape = [B, H, W, 2]

        H, W = uv.shape[1:-1]
        assert (H == W)

        # with torch.no_grad():
        mask = torch.greater(uv1_last, torch.ones_like(uv1[:, :, :, 2:]) * 1e-6) * \
               torch.greater_equal(uv[:, :, :, 0:1], torch.zeros_like(uv[:, :, :, 0:1])) * \
               torch.less(uv[:, :, :, 0:1], torch.ones_like(uv[:, :, :, 0:1]) * grd_W) * \
               torch.greater_equal(uv[:, :, :, 1:2], torch.zeros_like(uv[:, :, :, 1:2])) * \
               torch.less(uv[:, :, :, 1:2], torch.ones_like(uv[:, :, :, 1:2]) * grd_H)
        uv = uv * mask

        return uv, mask
        # return uv1

    def project_grd_to_map(self, grd_f, grd_c, shift_u, shift_v, heading, camera_k, satmap_sidelength, ori_grdH,
                           ori_grdW, require_jac=True):
        # inputs:
        #   grd_f: ground features: B,C,H,W
        #   shift: B, S, 2
        #   heading: heading angle: B,S
        #   camera_k: 3*3 K matrix of left color camera : B*3*3
        # return:
        #   grd_f_trans: B,S,E,C,satmap_sidelength,satmap_sidelength

        B, C, H, W = grd_f.size()

        XYZ_1 = self.sat2world(satmap_sidelength)  # [ sidelength,sidelength,4]

        if self.args.proj == 'geo' or self.args.proj == 'CrossAttn':
            uv, mask = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, camera_k, H, W, ori_grdH, ori_grdW)  # [B, S, E, H, W,2]
            # [B, H, W, 2], [B, H, W, 1]

        grd_f_trans, new_jac = grid_sample(grd_f, uv, None)
        # [B,C,sidelength,sidelength], [3, B, C, sidelength, sidelength]
        grd_f_trans = grd_f_trans * mask[:, None, :, :, 0]
        if grd_c is not None:
            grd_c_trans, _ = grid_sample(grd_c, uv)
            grd_c_trans = grd_c_trans * mask[:, None, :, :, 0]
        else:
            grd_c_trans = None


        return grd_f_trans, grd_c_trans, uv, mask

    def inplane_uv(self, ori_shift_u, ori_shift_v, ori_heading, satmap_sidelength):
        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength

        B = ori_heading.shape[0]
        shift_u_pixels = self.args.shift_range_lon * ori_shift_u / meter_per_pixel
        shift_v_pixels = self.args.shift_range_lat * ori_shift_v / meter_per_pixel
        T = torch.cat([-shift_u_pixels, shift_v_pixels], dim=-1)  # [B, 2]

        heading = ori_heading * self.args.rotation_range / 180 * np.pi
        cos = torch.cos(heading)
        sin = torch.sin(heading)
        R = torch.cat([cos, -sin, sin, cos], dim=-1).view(B, 2, 2)

        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        v, u = torch.meshgrid(i, j)  # i:h,j:w
        uv_2 = torch.stack([u, v], dim=-1).unsqueeze(dim=0).repeat(B, 1, 1, 1).float()  # [B, H, W, 2]
        uv_2 = uv_2 - satmap_sidelength / 2

        uv_1 = torch.einsum('bij, bhwj->bhwi', R, uv_2)
        uv_0 = uv_1 + T[:, None, None, :]  # [B, H, W, 2]

        uv = uv_0 + satmap_sidelength / 2
        return uv

    def Trans_update(self, shift_u, shift_v, heading, grd_feat_proj, sat_feat, level):
        B = shift_u.shape[0]
        grd_feat_norm = torch.norm(grd_feat_proj.reshape(B, -1), p=2, dim=-1)
        grd_feat_norm = torch.maximum(grd_feat_norm, 1e-6 * torch.ones_like(grd_feat_norm))
        grd_feat_proj = grd_feat_proj / grd_feat_norm[:, None, None, None]

        delta = self.TransRefine(grd_feat_proj, sat_feat, level)  # [B, 3]
        # print('=======================')
        # print('delta.shape: ', delta.shape)
        # print('shift_u.shape', shift_u.shape)
        # print('=======================')

        shift_u_new = shift_u + delta[:, 0:1]
        shift_v_new = shift_v + delta[:, 1:2]
        heading_new = heading + delta[:, 2:3]

        B = shift_u.shape[0]

        rand_u = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_v = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_u.requires_grad = True
        rand_v.requires_grad = True
        # shift_u_new = torch.where((shift_u_new > -2.5) & (shift_u_new < 2.5), shift_u_new, rand_u)
        # shift_v_new = torch.where((shift_v_new > -2.5) & (shift_v_new < 2.5), shift_v_new, rand_v)
        shift_u_new = torch.where((shift_u_new > -2) & (shift_u_new < 2), shift_u_new, rand_u)
        shift_v_new = torch.where((shift_v_new > -2) & (shift_v_new < 2), shift_v_new, rand_v)

        return shift_u_new, shift_v_new, heading_new

    def corr(self, sat_map, grd_img_left, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
             mode='train', file_name=None, gt_depth=None):
        '''
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            left_camera_k: [B, 3, 3]
            grd_img_left: [B, C, H, W]
            gt_shift_u: [B, 1] u->longitudinal
            gt_shift_v: [B, 1] v->lateral
            gt_heading: [B, 1] east as 0-degree
            mode:
            file_name:

        Returns:

        '''
        '''
        :param sat_map: [B, C, A, A] A--> sidelength
        :param left_camera_k: [B, 3, 3]
        :param grd_img_left: [B, C, H, W]
        :return:
        '''

        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)

        grd_feat_list, grd_conf_list = self.GrdFeatureNet(grd_img_left)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        corr_maps = []

        for level in range(len(sat_feat_list)):
            meter_per_pixel = self.meters_per_pixel[level]

            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]

            A = sat_feat.shape[-1]
            heading = gt_heading + np.random.uniform(- self.args.coe_heading_aug, self.args.coe_heading_aug)
            grd_feat_proj, _, grd_uv, mask = self.project_grd_to_map(
                grd_feat, None, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)

            crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
            crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
            g2s_feat = TF.center_crop(grd_feat_proj, [crop_H, crop_W])
            g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

            s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

            denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)  # [B, 4W]
            denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            corr = 2 - 2 * corr / denominator

            B, corr_H, corr_W = corr.shape

            corr_maps.append(corr)

            max_index = torch.argmin(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - corr_W / 2) * meter_per_pixel  # / self.args.shift_range_lon
            pred_v = -(max_index // corr_W - corr_H / 2) * meter_per_pixel  # / self.args.shift_range_lat

            cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)
            sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)

            pred_u1 = pred_u * cos + pred_v * sin
            pred_v1 = - pred_u * sin + pred_v * cos

            # corr0 = []
            # for b in range(B):
            #     corr0.append(F.conv2d(s_feat[b:b+1, :, :, :], g2s_feat[b:b+1, :, :, :]))  # [1, 1, H, W]
            # corr0 = torch.cat(corr0, dim=1)
            # print(torch.sum(torch.abs(corr0 - corr)))

        if mode == 'train':
            return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
        else:
            return pred_u1, pred_v1  # [B], [B]

    def triplet_loss(self, corr_maps, gt_shift_u, gt_shift_v, gt_heading):
        cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)
        sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)

        gt_delta_x = - gt_shift_u[:, 0] * self.args.shift_range_lon
        gt_delta_y = - gt_shift_v[:, 0] * self.args.shift_range_lat

        gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
        gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos

        losses = []
        for level in range(len(corr_maps)):
            meter_per_pixel = self.meters_per_pixel[level]

            corr = corr_maps[level]
            B, corr_H, corr_W = corr.shape

            w = torch.round(corr_W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)
            h = torch.round(corr_H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)

            pos = corr[range(B), h.long(), w.long()]  # [B]
            pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))
            # import pdb; pdb.set_trace()
            losses.append(loss)

        return torch.sum(torch.stack(losses, dim=0))

    def weak_supervise_loss(self, corr_maps):
        losses = []
        for corr in corr_maps:
            M, N, H, W = corr.shape
            assert M == N
            dis = torch.max(corr.reshape(M, N, -1), dim=-1)[0]
            pos = torch.diagonal(dis) # [M]
            pos_neg = pos.reshape(-1, 1) - dis
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
            losses.append(loss)

        return torch.sum(torch.stack(losses, dim=0))

    def CVattn_corr(self, sat_map, grd_img_left, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
                    mode='train'):
        '''
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            left_camera_k: [B, 3, 3]
            grd_img_left: [B, C, H, W]
            gt_shift_u: [B, 1] u->longitudinal
            gt_shift_v: [B, 1] v->lateral
            gt_heading: [B, 1] east as 0-degree
            mode:
            file_name:

        Returns:

        '''

        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)
        # if self.args.use_uncertainty:
        #     sat_uncer_list = self.uncertain_net(sat_feat_list)
        sat8, sat4, sat2 = sat_feat_list

        grd8, grd4, grd2 = self.GrdEnc(grd_img_left)
        # [H/8, W/8] [H/4, W/4] [H/2, W/2]
        grd_feat_list = self.GrdDec(grd8, grd4, grd2)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = gt_heading

        # heading = gt_heading + np.random.uniform(-0.1, 0.1)
        grd2sat8, _, u, mask = self.project_grd_to_map(
            grd_feat_list[0], None, shift_u, shift_v, heading, left_camera_k, sat8.shape[-1], ori_grdH, ori_grdW,
            require_jac=False)
        grd2sat4, _, _, _ = self.project_grd_to_map(
            grd_feat_list[1], None, shift_u, shift_v, heading, left_camera_k, sat4.shape[-1], ori_grdH, ori_grdW,
            require_jac=False)
        grd2sat2, _, _, _ = self.project_grd_to_map(
            grd_feat_list[2], None, shift_u, shift_v, heading, left_camera_k, sat2.shape[-1], ori_grdH, ori_grdW,
            require_jac=False)

        grd2sat8_attn = self.CVattn(grd2sat8, grd8, u, mask)
        grd2sat4_attn = grd2sat4 + self.Dec4(grd2sat8_attn, grd2sat4)
        grd2sat2_attn = grd2sat2 + self.Dec2(grd2sat4_attn, grd2sat2)

        grd_feat_list = [grd2sat8_attn, grd2sat4_attn, grd2sat2_attn]

        corr_maps = []

        for level in range(len(sat_feat_list)):
            meter_per_pixel = self.meters_per_pixel[level]

            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]

            A = sat_feat.shape[-1]

            crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
            crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
            g2s_feat = TF.center_crop(grd_feat, [crop_H, crop_W])
            g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

            s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

            denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)  # [B, 4W]
            # denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            # if self.args.use_uncertainty:
            #     denominator = torch.sum(denominator, dim=1) * TF.center_crop(sat_uncer_list[level],
            #                                                                  [corr.shape[1], corr.shape[2]])[:, 0]
            # else:
            denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            corr = 2 - 2 * corr / denominator

            B, corr_H, corr_W = corr.shape

            corr_maps.append(corr)

            max_index = torch.argmin(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - corr_W / 2) * meter_per_pixel  # / self.args.shift_range_lon
            pred_v = -(max_index // corr_W - corr_H / 2) * meter_per_pixel  # / self.args.shift_range_lat

            cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)
            sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)

            pred_u1 = pred_u * cos + pred_v * sin
            pred_v1 = - pred_u * sin + pred_v * cos

        if mode == 'train':
            return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
        else:
            return pred_u1, pred_v1  # [B], [B]

    def NeuralOptimizer(self, grd_feat_dict, sat_feat_dict, B, left_camera_k=None, ori_grdH=None, ori_grdW=None):

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
        heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)

        shift_us_all = []
        shift_vs_all = []
        headings_all = []
        for iter in range(self.N_iters):
            shift_us = []
            shift_vs = []
            headings = []
            for level in self.level:
                sat_feat = sat_feat_dict[level]
                grd_feat = grd_feat_dict[level]

                if self.args.stage == 0:
                    uv = self.inplane_uv(shift_u, shift_v, heading, sat_feat.shape[-1])
                    overhead_feat, _ = grid_sample(
                        grd_feat * self.masks[level].clone()[:, None, :, :].repeat(B, 1, 1, 1),
                        uv, jac=None)

                elif self.args.stage > 0:
                    A = sat_feat.shape[-1]
                    overhead_feat, _, _, _ = self.project_grd_to_map(
                        grd_feat, None, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)

                shift_u_new, shift_v_new, heading_new = self.Trans_update(
                    shift_u, shift_v, heading, overhead_feat, sat_feat, level)

                shift_us.append(shift_u_new[:, 0])  # [B]
                shift_vs.append(shift_v_new[:, 0])  # [B]
                headings.append(heading_new[:, 0])

                shift_u = shift_u_new.clone()
                shift_v = shift_v_new.clone()
                heading = heading_new.clone()

            shift_us_all.append(torch.stack(shift_us, dim=1))  # [B, Level]
            shift_vs_all.append(torch.stack(shift_vs, dim=1))  # [B, Level]
            headings_all.append(torch.stack(headings, dim=1))  # [B, Level]

        shift_lats = torch.stack(shift_vs_all, dim=1)  # [B, N_iters, Level]
        shift_lons = torch.stack(shift_us_all, dim=1)  # [B, N_iters, Level]
        thetas = torch.stack(headings_all, dim=1)  # [B, N_iters, Level]

        return shift_lats, shift_lons, thetas

    def forward(self, sat_align_cam, sat_map, grd_img_left, left_camera_k, gt_heading=None, gt_shift_u=None, gt_shift_v=None, train=False, loop=None, save_dir=None):
        '''
        rot_corr
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            left_camera_k: [B, 3, 3]
            grd_img_left: [B, C, H, W]
            gt_shift_u: [B, 1] u->longitudinal
            gt_shift_v: [B, 1] v->lateral
            gt_heading: [B, 1] east as 0-degree
            mode:
            file_name:

        Returns:

        '''

        # grd = transforms.ToPILImage()(grd_img_left[0])
        # grd.save('grd.png')
        # sat = transforms.ToPILImage()(sat_map[0])
        # sat.save('sat.png')
        # sat_align_cam_ = transforms.ToPILImage()(sat_align_cam[0])
        # sat_align_cam_.save('sat_align_cam.png')
        #
        # uv = self.inplane_uv(gt_shift_u, gt_shift_v, gt_heading, sat_map.shape[-1])
        # sat_align_cam_trans, _ = grid_sample(
        #     sat_align_cam,
        #     uv, jac=None)
        # sat_align_cam_trans = transforms.ToPILImage()(sat_align_cam_trans[0])
        # sat_align_cam_trans.save('sat_align_cam_trans.png')


        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        g2s_feat_dict = {}
        g2s_conf_dict = {}

        if self.args.stage == 0:
            sat_feat_dict, sat_conf_dict = self.SatFeatureNet(sat_map)
            over_feat_dict, over_conf_dict = self.SatFeatureNet(sat_align_cam)
            # not sure whether mask should be appliced at image level or feature level

            shift_lats, shift_lons, thetas = self.NeuralOptimizer(over_feat_dict, sat_feat_dict, B)

            for _, level in enumerate(self.level):
                meter_per_pixel = self.meters_per_pixel[level]
                sat_feat = sat_feat_dict[level]
                over_feat = over_feat_dict[level]
                over_conf = over_conf_dict[level]

                A = sat_feat.shape[-1]
                uv = self.inplane_uv(shift_u, shift_v, gt_heading, A)
                overhead_feat, _ = grid_sample(
                    over_feat * self.masks[level].clone()[:, None, :, :].repeat(B, 1, 1, 1),
                    uv, jac=None)
                overhead_conf, _ = grid_sample(
                    over_conf * self.masks[level].clone()[:, None, :, :].repeat(B, 1, 1, 1),
                    uv, jac=None
                )

                crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
                crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
                g2s_feat = TF.center_crop(overhead_feat, [crop_H, crop_W])
                overhead_conf = TF.center_crop(overhead_conf, [crop_H, crop_W])

                g2s_feat_dict[level] = g2s_feat
                g2s_conf_dict[level] = overhead_conf

            return sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, shift_lats, shift_lons, thetas, None

        else:

            with torch.no_grad():
                sat_feat_dict_forR, sat_uncer_dict_forR = self.SatFeatureNet(sat_map)
                grd_feat_dict_forR, grd_conf_dict_forR = self.SatFeatureNet(grd_img_left)

                if self.args.rotation_range > 0:
                    shift_lats, shift_lons, thetas = self.NeuralOptimizer(grd_feat_dict_forR, sat_feat_dict_forR, B,
                                                                          left_camera_k, ori_grdH, ori_grdW)
                    heading = thetas[:, -1, -1:].detach()
                else:
                    heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
                    shift_lats = None
                    shift_lons = None

            # ----------------- Translation Stage ---------------------------

            if self.args.share:
                grd_feat_dict_forT, grd_conf_dict_forT = self.FeatureForT(grd_img_left)
                sat_feat_dict_forT, sat_conf_dict_forT = self.FeatureForT(sat_map)
            else:
                grd_feat_dict_forT, grd_conf_dict_forT = self.GrdFeatureForT(grd_img_left)
                sat_feat_dict_forT, sat_conf_dict_forT = self.SatFeatureForT(sat_map)

            grd_uv_dict = {}
            mask_dict = {}
            for level in range(4):
                # meter_per_pixel = self.meters_per_pixel[level]
                sat_feat = sat_feat_dict_forT[level]
                grd_feat = grd_feat_dict_forT[level]

                A = sat_feat.shape[-1]
                grd_feat_proj, grd_conf_proj, grd_uv, mask = self.project_grd_to_map(
                    grd_feat, grd_conf_dict_forT[level], shift_u, shift_v, heading, left_camera_k, A, ori_grdH,
                    ori_grdW,
                    require_jac=False)

                g2s_feat_dict[level] = grd_feat_proj
                g2s_conf_dict[level] = grd_conf_proj
                grd_uv_dict[level] = grd_uv
                mask_dict[level] = mask

            if self.args.proj == 'CrossAttn':
                # import pdb; pdb.set_trace()
                grd2sat8_attn = self.CVattn(g2s_feat_dict[0], grd_feat_dict_forT[0], grd_uv_dict[0][..., 0], mask_dict[0])
                grd2sat4_attn = g2s_feat_dict[1] + self.Dec4(grd2sat8_attn, g2s_feat_dict[1])
                grd2sat2_attn = g2s_feat_dict[2] + self.Dec2(grd2sat4_attn, g2s_feat_dict[2])
                g2s_feat_dict[0] = grd2sat8_attn
                g2s_feat_dict[1] = grd2sat4_attn
                g2s_feat_dict[2] = grd2sat2_attn

            for _, level in enumerate(self.level):

                if self.args.visualize:

                    visualize_dir = os.path.join(save_dir, 'visualize_conf/')
                    if not os.path.exists(visualize_dir):
                        os.makedirs(visualize_dir)
                    for idx in range(B):
                        conf = grd_conf_dict_forT[level][idx][0].detach().cpu().numpy()
                        conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-7)
                        # conf[:conf.shape[0] // 2, :] = 0
                        img = cv2.applyColorMap(np.uint8(conf * 255),
                                                cv2.COLORMAP_JET)
                        img = cv2.resize(img, (1024, 256))
                        img = np.float32(img) / 255 + grd_img_left[idx].detach().cpu().numpy().transpose(1, 2, 0)
                        img = img / np.max(img) * 255
                        img = cv2.resize(np.uint8(img), (512, 256))

                        fig, ax = plt.subplots()
                        shw = ax.imshow(np.uint8(img[:, :, [2, 1, 0]]))

                        # plt.show()

                        norm = colors.Normalize(vmin=0, vmax=1)
                        shw.set_norm(norm)
                        shw.set_cmap('jet')
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(shw, cax=cax, orientation='vertical')

                        ax.axis('off')
                        plt.savefig(os.path.join(visualize_dir, 'grd_conf_' + str(level) + '_' + str(loop * B + idx) + '.png'),
                                    transparent=True, dpi=150, bbox_inches='tight', pad_inches=0)
                        plt.close()

                        transforms.ToPILImage()(grd_img_left[idx]).save(
                            visualize_dir + 'grd_img_' + str(loop * B + idx) + '.png')
                        transforms.ToPILImage()(sat_map[idx]).save(
                            visualize_dir + 'sat_img_' + str(loop * B + idx) + '.png')

                    from visualize_utils import features_to_RGB

                    grd_feat_proj_center, grd_conf_proj_center, _, _ = self.project_grd_to_map(
                        grd_feat_dict_forT[level], grd_conf_dict_forT[level], shift_u, shift_v, shift_u, left_camera_k,
                        sat_feat_dict_forT[level].shape[-1],
                        ori_grdH,
                        ori_grdW,
                        require_jac=False)

                    grd_feat_proj_gt, grd_conf_proj_gt, _, _ = self.project_grd_to_map(
                        grd_feat_dict_forT[level], grd_conf_dict_forT[level], gt_shift_u, gt_shift_v, gt_heading, left_camera_k,
                        sat_feat_dict_forT[level].shape[-1],
                        ori_grdH,
                        ori_grdW,
                        require_jac=False)

                    features_to_RGB(sat_feat_dict_forT[level], grd_feat_proj_center, grd_conf_proj_center,
                                    grd_feat_proj_gt, grd_conf_proj_gt, loop, level, visualize_dir)

                meter_per_pixel = self.meters_per_pixel[level]
                sat_feat = sat_feat_dict_forT[level]

                A = sat_feat.shape[-1]

                crop_H = int(A - 20 * 3 / meter_per_pixel)
                crop_W = int(A - 20 * 3 / meter_per_pixel)
                g2s_feat = TF.center_crop(g2s_feat_dict[level], [crop_H, crop_W])

                g2s_conf = TF.center_crop(g2s_conf_dict[level], [crop_H, crop_W])

                g2s_feat_dict[level] = g2s_feat
                g2s_conf_dict[level] = g2s_conf

            return sat_feat_dict_forT, sat_conf_dict_forT, g2s_feat_dict, g2s_conf_dict, shift_lats, shift_lons, thetas



def batch_wise_cross_corr(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args):
    '''
    compute corr_maps for training
    result corr_map has a shape of [M, N, H, W],
    M is the number of satellite images and N is the number of ground images
    '''

    levels = sorted([int(item) for item in args.level.split('_')])
    corr_maps = {}
    for _, level in enumerate(levels):
        sat_feat = sat_feat_dict[level]
        sat_conf = sat_conf_dict[level]
        g2s_feat = g2s_feat_dict[level]
        g2s_conf = g2s_conf_dict[level]

        B, C, crop_H, crop_W = g2s_feat.shape


        if args.ConfGrd > 0:

            if args.ConfSat > 0:

                # numerator
                signal = (sat_feat * sat_conf.pow(2)).repeat(1, B, 1, 1)   # [B(M), BC(NC), H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)

                # denominator
                denominator_sat = []
                sat_feat_conf_pow = (sat_feat * sat_conf).pow(2)
                g2s_conf_pow = g2s_conf.pow(2)
                for i in range(0, B):
                    denom_sat = torch.sum(F.conv2d(sat_feat_conf_pow[i, :, None, :, :], g2s_conf_pow), dim=0)
                    denominator_sat.append(denom_sat)
                denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))

                denominator_grd = []
                sat_conf_pow = sat_conf.pow(2)
                g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
                for i in range(0, B):
                    denom_grd = torch.sum(F.conv2d(sat_conf_pow[i:i+1, :, :, :].repeat(1, C, 1, 1), g2s_feat_conf_pow), dim=1)
                    denominator_grd.append(denom_grd)
                denominator_grd = torch.sqrt(torch.stack(denominator_grd, dim=0))

                # corr = corr / denominator_sat / denominator_grd

            else:

                # numerator
                signal = sat_feat.repeat(1, B, 1, 1)  # [B(M), BC(NC), H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)

                # denominator
                denominator_sat = []
                sat_feat_pow = (sat_feat).pow(2)
                g2s_conf_pow = g2s_conf.pow(2)
                for i in range(0, B):
                    denom_sat = torch.sum(F.conv2d(sat_feat_pow[i, :, None, :, :], g2s_conf_pow), dim=0)
                    denominator_sat.append(denom_sat)
                denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))  # [B (M), B (N), H, W]

                denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1) # [B]
                shape = denominator_sat.shape
                denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])

                # corr = corr / denominator_sat / denominator_grd

        else:

            signal = sat_feat.repeat(1, B, 1, 1)  # [B(M), BC(NC), H, W]
            kernel = g2s_feat
            corr = F.conv2d(signal, kernel, groups=B)

            denominator_sat = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1, keepdim=True))

            denom_grd = torch.linalg.norm((g2s_feat).reshape(B, -1), dim=-1)  # [B]
            shape = denominator_sat.shape
            denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])

            # denominator = corr / denominator_sat / denominator_grd

        denominator = denominator_sat * denominator_grd

        denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

        corr = 2 - 2 * corr / denominator  # [B, B, H, W]

        corr_maps[level] = corr

    return corr_maps


def weak_supervise_loss(corr_maps):
    '''
    triplet loss/ metric learning loss for self-supervision
    corr_maps: dict
    key -- level; value -- corr map
    '''
    losses = []
    for key, corr in corr_maps:
        M, N, H, W = corr.shape
        assert M == N
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis) # [M]
        pos_neg = pos.reshape(-1, 1) - dis
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
        losses.append(loss)

    return torch.mean(torch.stack(losses, dim=0))


def Weakly_supervised_loss_w_GPS_error(corr_maps, gt_shift_u, gt_shift_v, gt_heading, args, meter_per_pixels, GPS_error=5):
    '''
    GPS_error: scalar, in terms of meters
    '''
    matching_losses = []

    # ---------- preparing for GPS error Loss -------
    levels = [int(item) for item in args.level.split('_')]

    GPS_error_losses = []
    cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)

    gt_delta_x = - gt_shift_u[:, 0] * args.shift_range_lon
    gt_delta_y = - gt_shift_v[:, 0] * args.shift_range_lat

    gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
    gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos
    # ------------------------------------------------

    for _, level in enumerate(levels):
        corr = corr_maps[level]
        M, N, H, W = corr.shape
        assert M == N
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis) # [M]  # it is also the predicted distance
        pos_neg = pos.reshape(-1, 1) - dis
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
        matching_losses.append(loss)

        # ---------- preparing for GPS error Loss -------
        meter_per_pixel = meter_per_pixels[level]
        w = (torch.round(W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)).long() # [B]
        h = (torch.round(H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)).long() # [B]
        radius = int(np.ceil(GPS_error / meter_per_pixel))
        GPS_dis = []
        for b_idx in range(M):
            # GPS_dis.append(torch.min(corr[b_idx, b_idx, h[b_idx]-radius: h[b_idx]+radius, w[b_idx]-radius: w[b_idx]+radius]))
            start_h = torch.max(torch.tensor(0).long(), h[b_idx] - radius)
            end_h = torch.min(torch.tensor(corr.shape[2]).long(), h[b_idx] + radius)
            start_w = torch.max(torch.tensor(0).long(), w[b_idx] - radius)
            end_w = torch.min(torch.tensor(corr.shape[3]).long(), w[b_idx] + radius)
            GPS_dis.append(torch.min(
                corr[b_idx, b_idx, start_h: end_h, start_w: end_w]))
        GPS_error_losses.append(torch.abs(torch.stack(GPS_dis) - pos))

    return torch.mean(torch.stack(matching_losses, dim=0)), torch.mean(torch.stack(GPS_error_losses, dim=0))


def GT_triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading, args, meters_per_pixel):
    '''
    Used when GT GPS lables are highly reliable.
    This function does not handle the rotation issue.
    '''
    levels = [int(item) for item in args.level.split('_')]

    # cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    # sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    #
    # gt_delta_x = gt_shift_u[:, 0] * args.shift_range_lon
    # gt_delta_y = gt_shift_v[:, 0] * args.shift_range_lat
    #
    # gt_delta_x_rot = - gt_delta_x * cos - gt_delta_y * sin
    # gt_delta_y_rot = gt_delta_x * sin - gt_delta_y * cos

    cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)

    gt_delta_x = - gt_shift_u[:, 0] * args.shift_range_lon
    gt_delta_y = - gt_shift_v[:, 0] * args.shift_range_lat

    gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
    gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos

    losses = []
    # for level in range(len(corr_maps)):
    for _, level in enumerate(levels):
        corr = corr_maps[level]
        B, corr_H, corr_W = corr.shape

        meter_per_pixel = meters_per_pixel[level]

        w = torch.round(corr_W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)
        h = torch.round(corr_H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)

        pos = corr[range(B), h.long(), w.long()]  # [B]
        pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))

        losses.append(loss)

    return torch.sum(torch.stack(losses, dim=0))


def corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args, meter_per_pixels, gt_heading):
    '''
    to be used during inference
    '''

    level = max([int(item) for item in args.level.split('_')])
    meter_per_pixel = meter_per_pixels[level]

    sat_feat = sat_feat_dict[level]
    sat_conf = sat_conf_dict[level]
    g2s_feat = g2s_feat_dict[level]
    g2s_conf = g2s_conf_dict[level]

    B, C, crop_H, crop_W = g2s_feat.shape
    A = sat_feat.shape[2]

    if args.ConfGrd > 0:

        if args.ConfSat > 0:

            # numerator
            signal = (sat_feat * sat_conf.pow(2)).reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            kernel = g2s_feat * g2s_conf.pow(2)
            corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

            # denominator
            sat_feat_conf_pow = (sat_feat * sat_conf).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
            g2s_conf_pow = g2s_conf.pow(2)
            denominator_sat = F.conv2d(sat_feat_conf_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

            sat_conf_pow = sat_conf.pow(2).repeat(1, C, 1, 1).reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
            denominator_grd = F.conv2d(sat_conf_pow, g2s_feat_conf_pow, groups=B)[0]  # [B, H, W]
            denominator_grd = torch.sqrt(denominator_grd)

        else:

            # numerator
            signal = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            kernel = g2s_feat * g2s_conf.pow(2)
            corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

            # denominator
            sat_feat_pow = (sat_feat).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
            g2s_conf_pow = g2s_conf.pow(2)
            denominator_sat = F.conv2d(sat_feat_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

            denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1)  # [B]
            shape = denominator_sat.shape
            denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])

            # corr = corr / denominator_sat / denominator_grd

    else:

        signal = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
        kernel = g2s_feat
        corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

        denominator_sat = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
        denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))

        denom_grd = torch.linalg.norm(g2s_feat.reshape(B, -1), dim=-1)  # [B]
        shape = denominator_sat.shape
        denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])
        # denominator = corr / denominator_sat / denominator_grd

    denominator = denominator_sat * denominator_grd

    denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

    corr = corr / denominator  # [B, H, W]

    corr_H = int(args.shift_range_lat * 3 / meter_per_pixel)
    corr_W = int(args.shift_range_lon * 3 / meter_per_pixel)

    corr = TF.center_crop(corr[:, None], [corr_H, corr_W])[:, 0]

    B, corr_H, corr_W = corr.shape

    max_index = torch.argmax(corr.reshape(B, -1), dim=1)

    if args.visualize:
        pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * np.power(2, 3 - level)
        pred_v = (max_index // corr_W - corr_H / 2 + 0.5) * np.power(2, 3 - level)
        return pred_u, pred_v, corr

    else:

        pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * meter_per_pixel  # / self.args.shift_range_lon
        pred_v = -(max_index // corr_W - corr_H / 2 + 0.5) * meter_per_pixel  # / self.args.shift_range_lat

        cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
        sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)

        pred_u1 = pred_u * cos + pred_v * sin
        pred_v1 = - pred_u * sin + pred_v * cos

        return pred_u1, pred_v1, corr



def corr_for_accurate_translation_supervision(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args,
                                              sat_uncer_dict=None):
    levels = [int(item) for item in args.level.split('_')]

    corr_maps = {}
    for level in levels:

        sat_feat = sat_feat_dict[level]
        sat_conf = sat_conf_dict[level]
        g2s_feat = g2s_feat_dict[level]
        g2s_conf = g2s_conf_dict[level]

        B, C, crop_H, crop_W = g2s_feat.shape
        A = sat_feat.shape[2]

        # s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
        # corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]
        #
        # if args.ConfGrd > 0:
        #     denominator = F.conv2d(sat_feat.pow(2).transpose(0, 1), g2s_conf.pow(2), groups=B).transpose(0, 1)
        # else:
        #     denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)

        if args.ConfGrd > 0:

            if args.ConfSat > 0:

                # numerator
                signal = (sat_feat * sat_conf.pow(2)).reshape(1, -1, A, A)    # [B, C, H, W]->[1, B*C, H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)[0]   # [B, H, W]

                # denominator
                sat_feat_conf_pow = (sat_feat * sat_conf).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
                g2s_conf_pow = g2s_conf.pow(2)
                denominator_sat = F.conv2d(sat_feat_conf_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
                denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

                sat_conf_pow = sat_conf.pow(2).repeat(1, C, 1, 1).reshape(1, -1, A, A)    # [B, C, H, W]->[1, B*C, H, W]
                g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
                denominator_grd = F.conv2d(sat_conf_pow, g2s_feat_conf_pow, groups=B)[0]  # [B, H, W]
                denominator_grd = torch.sqrt(denominator_grd)

            else:

                # numerator
                signal = sat_feat.reshape(1, -1, A, A)    # [B, C, H, W]->[1, B*C, H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)[0]   # [B, H, W]

                # denominator
                sat_feat_pow = (sat_feat).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
                g2s_conf_pow = g2s_conf.pow(2)
                denominator_sat = F.conv2d(sat_feat_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
                denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

                denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1) # [B]
                shape = denominator_sat.shape
                denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])

                # corr = corr / denominator_sat / denominator_grd

        else:

            signal = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            kernel = g2s_feat
            corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

            denominator_sat = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))

            denom_grd = torch.linalg.norm((g2s_feat).reshape(B, -1), dim=-1)  # [B]
            shape = denominator_sat.shape
            denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])
            # denominator = corr / denominator_sat / denominator_grd

        denominator = denominator_sat * denominator_grd

        # if args.use_uncertainty:
        #     denominator = denominator * TF.center_crop(sat_uncer_dict[level], [corr.shape[1], corr.shape[2]])[:, 0]

        denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

        corr = corr / denominator

        corr_maps[level] = 2 - 2 * corr

    return corr_maps




def loss_func(shift_lats, shift_lons, thetas,
              gt_shift_lat, gt_shift_lon, gt_theta,
              coe_shift_lat=100, coe_shift_lon=100, coe_theta=100):
    '''
    Args:
        loss_method:
        ref_feat_list:
        pred_feat_dict:
        gt_feat_dict:
        shift_lats: [B, N_iters, Level]
        shift_lons: [B, N_iters, Level]
        thetas: [B, N_iters, Level]
        gt_shift_lat: [B]
        gt_shift_lon: [B]
        gt_theta: [B]
        pred_uv_dict:
        gt_uv_dict:
        coe_shift_lat:
        coe_shift_lon:
        coe_theta:
        coe_L1:
        coe_L2:
        coe_L3:
        coe_L4:

    Returns:

    '''

    shift_lat_delta0 = torch.abs(shift_lats - gt_shift_lat[:, None, None])  # [B, N_iters, Level]
    shift_lon_delta0 = torch.abs(shift_lons - gt_shift_lon[:, None, None])  # [B, N_iters, Level]
    thetas_delta0 = torch.abs(thetas - gt_theta[:, None, None])  # [B, N_iters, level]

    shift_lat_delta = torch.mean(shift_lat_delta0, dim=0)  # [N_iters, Level]
    shift_lon_delta = torch.mean(shift_lon_delta0, dim=0)  # [N_iters, Level]
    thetas_delta = torch.mean(thetas_delta0, dim=0)  # [N_iters, level]

    shift_lat_decrease = shift_lat_delta[0, 0] - shift_lat_delta[-1, -1]  # scalar
    shift_lon_decrease = shift_lon_delta[0, 0] - shift_lon_delta[-1, -1]  # scalar
    thetas_decrease = thetas_delta[0, 0] - thetas_delta[-1, -1]  # scalar

    losses = coe_shift_lat * shift_lat_delta + coe_shift_lon * shift_lon_delta + coe_theta * thetas_delta  # [N_iters, level]
    loss_decrease = losses[0, 0] - losses[-1, -1]  # scalar
    loss = torch.mean(losses)  # mean or sum
    loss_last = losses[-1]

    return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
        shift_lat_delta[-1, -1], shift_lon_delta[-1, -1], thetas_delta[-1, -1]

