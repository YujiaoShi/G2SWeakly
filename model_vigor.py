import time

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
import utils
import os
import torchvision.transforms.functional as TF

# from GRU1 import ElevationEsitimate,VisibilityEsitimate,VisibilityEsitimate2,GRUFuse
# from VGG import VGGUnet, VGGUnet_G2S
from VGG import VGGUnet, VGGUnet_G2S, Encoder, Decoder, Decoder2, Decoder4, VGGUnetTwoDec
from jacobian import grid_sample

# from models_ford import loss_func
from RNNs import NNrefine, Uncertainty
from swin_transformer import TransOptimizerS2GP_V1, TransOptimizerG2SP_V1
from swin_transformer_cross import TransOptimizerG2SP, TransOptimizerG2SPV2, SwinTransformerSelf
from cross_attention import CrossViewAttention
from RNNs import NNrefine, Uncertainty, VisibilityMask
import cv2

EPS = utils.EPS


class ModelVIGOR(nn.Module):
    def __init__(self, args, device=None):  # device='cuda:0',
        super(ModelVIGOR, self).__init__()

        self.args = args
        self.device = device

        self.level = sorted([int(item) for item in args.level.split('_')])
        self.N_iters = args.N_iters
        self.channels = ([int(item) for item in self.args.channels.split('_')])

        self.SatFeatureNet = VGGUnet(self.level, self.channels)
        self.GrdFeatureNet = VGGUnet(self.level, self.channels)

        if self.args.share:
            self.FeatureForT = VGGUnet(self.level, self.channels)
        else:
            self.GrdFeatureForT = VGGUnet(self.level, self.channels)
            self.SatFeatureForT = VGGUnet(self.level, self.channels)

        self.TransRefine = TransOptimizerG2SP_V1(self.channels)

        if self.args.ConfGrd == 2:
            self.VisMask = VisibilityMask(dims=[256] + self.channels)

        self.coe_R = nn.Parameter(torch.tensor(-5., dtype=torch.float32), requires_grad=True)
        self.coe_T = nn.Parameter(torch.tensor(-3., dtype=torch.float32), requires_grad=True)

        if self.args.use_uncertainty:
            self.uncertain_net = Uncertainty(self.channels)

        self.grd_height = -2

        torch.autograd.set_detect_anomaly(True)


    def sat2grd_uv(self, rot, shift_u, shift_v, level, H, W, meter_per_pixel):
        '''
        rot.shape = [B]
        shift_u.shape = [B]
        shift_v.shape = [B]
        H: scalar  height of grd feature map, from which projection is conducted
        W: scalar  width of grd feature map, from which projection is conducted
        '''

        B = shift_u.shape[0]

        # shift_u = shift_u / np.power(2, 3 - level)
        # shift_v = shift_v / np.power(2, 3 - level)

        S = 512 / np.power(2, 3 - level)
        shift_u = shift_u * S / 4
        shift_v = shift_v * S / 4

        # shift_u = shift_u / 512 * S
        # shift_v = shift_v / 512 * S

        ii, jj = torch.meshgrid(torch.arange(0, S, dtype=torch.float32, device=shift_u.device),
                                torch.arange(0, S, dtype=torch.float32, device=shift_u.device))
        ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] v dimension
        jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] u dimension

        radius = torch.sqrt((ii-(S/2-0.5 + shift_v.reshape(-1, 1, 1)))**2 + (jj-(S/2-0.5 + shift_u.reshape(-1, 1, 1)))**2)

        theta = torch.atan2(ii - (S / 2 - 0.5 + shift_v.reshape(-1, 1, 1)), jj - (S / 2 - 0.5 + shift_u.reshape(-1, 1, 1)))
        theta = (-np.pi / 2 + (theta) % (2 * np.pi)) % (2 * np.pi)
        theta = (theta + rot[:, None, None] * self.args.rotation_range / 180 * np.pi) % (2 * np.pi)

        theta = theta / 2 / np.pi * W

        # meter_per_pixel = self.meter_per_pixel_dict[city] * 512 / S
        meter_per_pixel = meter_per_pixel * np.power(2, 3-level)
        phimin = torch.atan2(radius * meter_per_pixel[:, None, None], torch.tensor(self.grd_height))
        phimin = phimin / np.pi * H

        uv = torch.stack([theta, phimin], dim=-1)

        return uv

    def project_grd_to_map(self, grd_f, grd_c, rot, shift_u, shift_v, level, meter_per_pixel):
        '''
        grd_f.shape = [B, C, H, W]
        shift_u.shape = [B]
        shift_v.shape = [B]
        '''
        B, C, H, W = grd_f.size()
        uv = self.sat2grd_uv(rot, shift_u, shift_v, level, H, W, meter_per_pixel)  # [B, S, S, 2]
        grd_f_trans, _ = grid_sample(grd_f, uv)
        if grd_c is not None:
            grd_c_trans, _ = grid_sample(grd_c, uv)
        else:
            grd_c_trans = None
        return grd_f_trans, grd_c_trans, uv

    def forward_projImg(self, sat_map, grd_img_left, meter_per_pixel, gt_shift_u=None, gt_shift_v=None, gt_rot=None, mode='train'):

        '''
        :param sat_map: [B, C, A, A] A--> sidelength
        :param left_camera_k: [B, 3, 3]
        :param grd_img_left: [B, C, H, W]
        :return:
        '''

        B, _, ori_grdH, ori_grdW = grd_img_left.shape
        A = sat_map.shape[-1]
        sat_align_cam_trans, _, _ = self.project_grd_to_map(
            grd_img_left, None, gt_rot, gt_shift_u, gt_shift_v, level=3, meter_per_pixel=meter_per_pixel)
        # print("sat_align_cam_trans: ",sat_align_cam_trans.size)

        grd_img = transforms.ToPILImage()(sat_align_cam_trans[0])
        grd_img.save('./grd2sat.png')
        sat_align_cam = transforms.ToPILImage()(grd_img_left[0])
        sat_align_cam.save('./grd.png')
        sat = transforms.ToPILImage()(sat_map[0])
        sat.save('./sat.png')

        print('done')

    def Trans_update(self, shift_u, shift_v, rot, grd_feat_proj, sat_feat):
        B = shift_u.shape[0]
        grd_feat_norm = torch.norm(grd_feat_proj.reshape(B, -1), p=2, dim=-1)
        grd_feat_norm = torch.maximum(grd_feat_norm, 1e-6 * torch.ones_like(grd_feat_norm))
        grd_feat_proj = grd_feat_proj / grd_feat_norm[:, None, None, None]

        delta = self.TransRefine(grd_feat_proj, sat_feat)  # [B, 3]

        shift_u_new = shift_u + delta[:, 0]
        shift_v_new = shift_v + delta[:, 1]
        heading_new = rot + delta[:, 2]

        B = shift_u.shape[0]

        rand_u = torch.distributions.uniform.Uniform(-1, 1).sample([B]).to(shift_u.device)
        rand_v = torch.distributions.uniform.Uniform(-1, 1).sample([B]).to(shift_u.device)
        rand_u.requires_grad = True
        rand_v.requires_grad = True
        # shift_u_new = torch.where((shift_u_new > -2.5) & (shift_u_new < 2.5), shift_u_new, rand_u)
        # shift_v_new = torch.where((shift_v_new > -2.5) & (shift_v_new < 2.5), shift_v_new, rand_v)
        shift_u_new = torch.where((shift_u_new > -2) & (shift_u_new < 2), shift_u_new, rand_u)
        shift_v_new = torch.where((shift_v_new > -2) & (shift_v_new < 2), shift_v_new, rand_v)

        return shift_u_new, shift_v_new, heading_new

    def forward2DoF(self, sat, grd, meter_per_pixel, gt_rot=None, loop=None, save_dir=None):

        B = sat.shape[0]

        if self.args.share:
            sat_feat_dict, sat_conf_dict = self.FeatureForT(sat)
            grd_feat_dict, grd_conf_dict = self.FeatureForT(grd)

        else:
            sat_feat_dict, sat_conf_dict = self.SatFeatureNet(sat)
            grd_feat_dict, grd_conf_dict = self.GrdFeatureNet(grd)

        if self.args.use_uncertainty:
            sat_uncer_dict = self.uncertain_net(sat_feat_dict)
        else:
            sat_uncer_dict = {}
            for level in range(3):
                sat_uncer_dict[level] = None

        shift_u = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat.device)
        shift_v = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat.device)

        g2s_feat_dict = {}
        g2s_conf_dict = {}

        for _, level in enumerate(self.level):

            if self.args.visualize:
                visualize_dir = os.path.join(save_dir, 'visualize_conf/')
                if not os.path.exists(visualize_dir):
                    os.makedirs(visualize_dir)
                for idx in range(B):
                    conf = grd_conf_dict[level][idx][0].detach().cpu().numpy()
                    conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-7)
                    conf[:conf.shape[0] // 2, :] = 0
                    img = cv2.applyColorMap(np.uint8(conf * 255),
                                            cv2.COLORMAP_JET)
                    img = cv2.resize(img, (640, 320))
                    img = np.float32(img) / 255 + grd[idx].detach().cpu().numpy().transpose(1, 2, 0)
                    img = img / np.max(img) * 255
                    cv2.imwrite(visualize_dir + 'grd_conf_' + str(level) + '_' + str(loop * B + idx) + '.png',
                                img.astype(np.uint8))

                    transforms.ToPILImage()(grd[idx]).save(
                        visualize_dir + 'grd_img_' + str(loop * B + idx) + '.png')
                    transforms.ToPILImage()(sat[idx]).save(
                        visualize_dir + 'sat_img_' + str(loop * B + idx) + '.png')

            sat_feat = sat_feat_dict[level]
            grd_feat = grd_feat_dict[level]

            A = sat_feat.shape[-1]

            grd_feat_proj, grd_conf_proj, grd_uv = self.project_grd_to_map(
                grd_feat, grd_conf_dict[level], gt_rot, shift_u, shift_v, level, meter_per_pixel)

            crop_H = int(A * 0.4)
            crop_W = int(A * 0.4)
            g2s_feat = TF.center_crop(grd_feat_proj, [crop_H, crop_W])
            # if self.args.ConfGrd == 0:
            #     g2s_conf = TF.center_crop(grd_conf_proj, [crop_H, crop_W])
            #     g2s_feat = F.normalize((g2s_feat).reshape(B, -1)).reshape(B, -1, crop_H, crop_W)
            # elif self.args.ConfGrd == 1:
            #     g2s_conf = TF.center_crop(grd_conf_proj, [crop_H, crop_W])
            #     g2s_feat = F.normalize((g2s_feat * g2s_conf).reshape(B, -1)).reshape(B, -1, crop_H, crop_W)
            # elif self.args.ConfGrd == 2:
            #     vis_mask = self.VisMask(grd_conf_proj, grd_uv.permute(0, 3, 1, 2), grd_feat_proj)
            #     g2s_conf = TF.center_crop(vis_mask, [crop_H, crop_W])
            #     g2s_feat = F.normalize((g2s_feat * g2s_conf).reshape(B, -1)).reshape(B, -1, crop_H, crop_W)
            #
            # if self.args.ConfSat == 0:
            #     sat_feat_dict[level] = sat_feat
            # else:
            #     sat_feat_dict[level] = F.normalize((sat_feat * sat_conf_dict[level]).reshape(B, -1))\
            #         .reshape(B, *sat_feat.shape[1:])

            if self.args.ConfGrd == 2:
                vis_mask = self.VisMask(grd_conf_proj, grd_uv.permute(0, 3, 1, 2), grd_feat_proj)
                g2s_conf = TF.center_crop(vis_mask, [crop_H, crop_W])
            else:
                g2s_conf = TF.center_crop(grd_conf_proj, [crop_H, crop_W])

            g2s_feat_dict[level] = g2s_feat
            g2s_conf_dict[level] = g2s_conf
            # sat_uncer_dict[level] = sat_uncer_list[idx] if self.args.use_uncertainty else None

        return sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, sat_uncer_dict

    def inplane_uv(self, rot, shift_u, shift_v, level):
        '''
        rot: [B]
        shift_u: [B]
        shift_v: [B]
        level: scalar
        meter_per_pixel: [B]
        '''

        B = shift_u.shape[0]

        # shift_u = shift_u / np.power(2, 3 - level)
        # shift_v = shift_v / np.power(2, 3 - level)

        S = 512 / np.power(2, 3 - level)
        shift_u = shift_u * S / 4
        shift_v = shift_v * S / 4

        T = torch.stack([-shift_u, -shift_v], dim=-1)  # [B, 2]

        rot_radian = rot * self.args.rotation_range / 180 * np.pi
        cos = torch.cos(rot_radian)
        sin = torch.sin(rot_radian)
        R = torch.stack([cos, -sin, sin, cos], dim=-1).view(B, 2, 2)
        # print('in  rot.dtype', rot.dtype)
        # print('R.dtype', R.dtype)

        i = j = torch.arange(0, S).to(self.device)
        v, u = torch.meshgrid(i, j)  # i:h,j:w
        uv_2 = torch.stack([u, v], dim=-1).unsqueeze(dim=0).repeat(B, 1, 1, 1).float()  # [B, H, W, 2]
        uv_2 = uv_2 - S / 2

        return torch.einsum('bij, bhwj->bhwi', R, uv_2 + T[:, None, None, :]) + S/2

        # uv_1 = torch.einsum('bij, bhwj->bhwi', R, uv_2)
        # uv_0 = uv_1 + T[:, None, None, :]  # [B, H, W, 2]
        # uv = uv_0 + S / 2
        #
        # return uv

    def NeuralOptimizer(self, grd_feat_dict, sat_feat_dict, B, meter_per_pixel, stage=None):

        shift_u = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=self.device)
        shift_v = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=self.device)
        rot = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=self.device)

        shift_us_all = []
        shift_vs_all = []
        headings_all = []
        for iter in range(self.N_iters):
            shift_us = []
            shift_vs = []
            headings = []
            for level in self.level:
                # print(iter, level)
                sat_feat = sat_feat_dict[level]
                grd_feat = grd_feat_dict[level]

                if stage == 0:
                    # print('out rot.dtype', rot.dtype)
                    uv = self.inplane_uv(rot, shift_u, shift_v, level)
                    overhead_feat, _ = grid_sample(grd_feat, uv, jac=None)

                elif stage == 1:
                    sat_feat, _, _ = self.project_grd_to_map(
                        sat_feat, None,
                        torch.zeros_like(rot),
                        torch.zeros_like(shift_u),
                        torch.zeros_like(shift_v), level, meter_per_pixel)

                    overhead_feat, _, _ = self.project_grd_to_map(
                        grd_feat, None, rot, shift_u, shift_v, level, meter_per_pixel)

                elif stage > 1:
                    overhead_feat, _, _ = self.project_grd_to_map(
                        grd_feat, None, rot, shift_u, shift_v, level, meter_per_pixel)

                shift_u_new, shift_v_new, rot_new = self.Trans_update(
                    shift_u, shift_v, rot, overhead_feat, sat_feat)

                # print('rot_new.dtype', rot_new.dtype)

                shift_us.append(shift_u_new)  # [B]
                shift_vs.append(shift_v_new)  # [B]
                headings.append(rot_new)

                shift_u = shift_u_new.clone()
                shift_v = shift_v_new.clone()
                rot = rot_new.clone()
                # print('rot.dtype', rot.dtype)

            shift_us_all.append(torch.stack(shift_us, dim=1))  # [B, Level]
            shift_vs_all.append(torch.stack(shift_vs, dim=1))  # [B, Level]
            headings_all.append(torch.stack(headings, dim=1))  # [B, Level]

        shift_lats = torch.stack(shift_vs_all, dim=1)  # [B, N_iters, Level]
        shift_lons = torch.stack(shift_us_all, dim=1)  # [B, N_iters, Level]
        thetas = torch.stack(headings_all, dim=1)  # [B, N_iters, Level]

        return shift_lats, shift_lons, thetas

    def forward3DoF(self, sat_rt, sat, grd, meter_per_pixel, gt_rot=None, gt_shift_u=None, gt_shift_v=None, stage=None):
        '''
        sat_rt: sat mimic grd [B, C, S, S]
        # grd_align: grd mimic sat [B, C, H, W]
        sat: [B, C, S, S]
        grd: [B, C, H, W]
        meter_per_pixel: [B]
        gt_rot: [B]
        '''

        # uv = self.inplane_uv(gt_rot, gt_shift_u, gt_shift_v, level=3)
        # sat_align_cam_trans, _ = grid_sample(
        #     sat_rt,
        #     uv, jac=None)
        #
        # for b_idx in range(sat.shape[0]):
        #     transforms.ToPILImage()(sat_align_cam_trans[b_idx]).save('sat_transform' + str(b_idx) + '.png')
        #     transforms.ToPILImage()(sat[b_idx]).save('sat' + str(b_idx) + '.png')
        #     transforms.ToPILImage()(sat_rt[b_idx]).save('sat_rt' + str(b_idx) + '.png')

        B = sat.shape[0]

        shift_u = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat.device)
        shift_v = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat.device)

        g2s_feat_dict = {}
        g2s_conf_dict = {}

        if stage == 0:
            # start_time = time.time()
            sat_feat_dict, sat_conf_dict = self.SatFeatureNet(sat)
            over_feat_dict, over_conf_dict = self.SatFeatureNet(sat_rt)
            # feature_extraction_end = time.time()
            # print('feature extract time: ', feature_extraction_end-start_time)

            shift_lats, shift_lons, thetas = self.NeuralOptimizer(over_feat_dict, sat_feat_dict, B, meter_per_pixel, stage=0)
            # time_end_NO = time.time()
            # print('neural optimizer time: ', time_end_NO - feature_extraction_end)

            for _, level in enumerate(self.level):

                sat_feat = sat_feat_dict[level]
                over_feat = over_feat_dict[level]

                A = sat_feat.shape[-1]
                uv = self.inplane_uv(gt_rot, shift_u, shift_v, level)
                overhead_feat, _ = grid_sample(over_feat, uv, jac=None)

                crop_H = int(A // 2)
                crop_W = int(A // 2)
                g2s_feat = TF.center_crop(overhead_feat, [crop_H, crop_W])

                g2s_feat_dict[level] = g2s_feat
                g2s_conf_dict[level] = None

            # print('corr time:             ', time.time() - time_end_NO)
            return sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, shift_lats, shift_lons, thetas

        elif stage == 1:

            assert self.args.rotation_range > 0
            sat_feat_dict, sat_conf_dict = self.GrdFeatureNet(sat)
            over_feat_dict, over_conf_dict = self.GrdFeatureNet(grd)
            shift_lats, shift_lons, thetas = self.NeuralOptimizer(over_feat_dict, sat_feat_dict, B, meter_per_pixel, stage=1)

            # ----------------- Second Stage ---------------------------

            for _, level in enumerate(self.level):
                # meter_per_pixel = self.meters_per_pixel[level]
                sat_feat = sat_feat_dict[level]
                over_feat = over_feat_dict[level]

                A = sat_feat.shape[-1]

                uv = self.inplane_uv(gt_rot, shift_u, shift_v, level)
                sat_feat, _ = grid_sample(sat_feat, uv, jac=None)
                overhead_feat, _ = grid_sample(over_feat, uv, jac=None)

                crop_H = int(A // 2)
                crop_W = int(A // 2)
                g2s_feat = TF.center_crop(overhead_feat, [crop_H, crop_W])

                g2s_feat_dict[level] = g2s_feat
                g2s_conf_dict[level] = None
                sat_feat_dict[level] = sat_feat

                # print('corr time:             ', time.time() - time_end_NO)
            return sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, shift_lats, shift_lons, thetas

        elif stage == 2:

            with torch.no_grad():
                sat_feat_dict_forR, sat_conf_dict_forR = self.SatFeatureNet(sat)
                grd_feat_dict_forR, grd_conf_dict_forR = self.SatFeatureNet(grd)
                if self.args.rotation_range > 0:
                    shift_lats, shift_lons, thetas = self.NeuralOptimizer(grd_feat_dict_forR, sat_feat_dict_forR, B,
                                                                          meter_per_pixel, stage=2)
                    pred_rot = thetas[:, -1, -1].detach()
                else:
                    thetas = torch.zeros([B, self.N_iters, len(self.level)], dtype=torch.float32, requires_grad=True, device=sat.device)
                    pred_rot = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat.device)
                    shift_lats = None
                    shift_lons = None

            # ----------------- Second Stage ---------------------------
            if self.args.share:
                grd_feat_dict_forT, grd_conf_dict_forT = self.FeatureForT(grd)
                sat_feat_dict_forT, sat_conf_dict_forT = self.FeatureForT(sat)
            else:
                grd_feat_dict_forT, grd_conf_dict_forT = self.GrdFeatureForT(grd)
                sat_feat_dict_forT, sat_conf_dict_forT = self.SatFeatureForT(sat)
            # sat_feat_dict_forT = {}
            # sat_uncer_dict_forT = {}
            for _, level in enumerate(self.level):
                # meter_per_pixel = self.meters_per_pixel[level]
                sat_feat = sat_feat_dict_forT[level]
                grd_feat = grd_feat_dict_forT[level]

                A = sat_feat.shape[-1]

                grd_feat_proj, grd_conf_proj, grd_uv = self.project_grd_to_map(
                    grd_feat, grd_conf_dict_forT[level], pred_rot, shift_u, shift_v, level, meter_per_pixel)

                crop_H = int(A // 2)
                crop_W = int(A // 2)
                g2s_feat = TF.center_crop(grd_feat_proj, [crop_H, crop_W])
                # if self.args.ConfGrd == 0:
                #     g2s_conf = TF.center_crop(grd_conf_proj, [crop_H, crop_W])
                #     g2s_feat = F.normalize((g2s_feat).reshape(B, -1)).reshape(B, -1, crop_H, crop_W)
                # elif self.args.ConfGrd == 1:
                #     g2s_conf = TF.center_crop(grd_conf_proj, [crop_H, crop_W])
                #     g2s_feat = F.normalize((g2s_feat * g2s_conf).reshape(B, -1)).reshape(B, -1, crop_H, crop_W)
                # elif self.args.ConfGrd == 2:
                #     vis_mask = self.VisMask(grd_conf_proj, grd_uv.permute(0, 3, 1, 2), grd_feat_proj)
                #     g2s_conf = TF.center_crop(vis_mask, [crop_H, crop_W])
                #     g2s_feat = F.normalize((g2s_feat * g2s_conf).reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

                if self.args.ConfGrd == 2:
                    vis_mask = self.VisMask(grd_conf_proj, grd_uv.permute(0, 3, 1, 2), grd_feat_proj)
                    g2s_conf = TF.center_crop(vis_mask, [crop_H, crop_W])
                else:
                    g2s_conf = TF.center_crop(grd_conf_proj, [crop_H, crop_W])

                # if self.args.ConfSat == 0:
                #     sat_feat_dict_forT[level] = sat_feat
                # else:
                #     sat_feat_dict_forT[level] = F.normalize((sat_feat * sat_conf_dict_forT[level]).reshape(B, -1))\
                #         .reshape(B, *sat_feat.shape[1:])

                g2s_feat_dict[level] = g2s_feat
                g2s_conf_dict[level] = g2s_conf

            return sat_feat_dict_forT, sat_conf_dict_forT, g2s_feat_dict, g2s_conf_dict, shift_lats, shift_lons, thetas

        elif stage > 2:

            # if self.args.stage == 1:
            #     with torch.no_grad():
            #         sat_feat_dict_forR, sat_conf_dict_forR = self.SatFeatureNet(sat)
            #         grd_feat_dict_forR, grd_conf_dict_forR = self.SatFeatureNet(grd)
            #         if self.args.rotation_range > 0:
            #             shift_lats, shift_lons, thetas = self.NeuralOptimizer(grd_feat_dict_forR, sat_feat_dict_forR, B,
            #                                                                   meter_per_pixel)
            #             pred_rot = thetas[:, -1, -1].detach()
            #         else:
            #             pred_rot = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat.device)
            #             shift_lats = None
            #             shift_lons = None
            # else:

            if self.args.rotation_range > 0:
                sat_feat_dict_forR, sat_conf_dict_forR = self.SatFeatureNet(sat)
                grd_feat_dict_forR, grd_conf_dict_forR = self.GrdFeatureNet(grd)
                shift_lats, shift_lons, thetas = self.NeuralOptimizer(grd_feat_dict_forR, sat_feat_dict_forR, B,
                                                                      meter_per_pixel, stage=2)
                pred_rot = thetas[:, -1, -1]
            else:
                pred_rot = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat.device)
                shift_lats = None
                shift_lons = None

            # ----------------- Second Stage ---------------------------
            if self.args.share:
                grd_feat_dict_forT, grd_conf_dict_forT = self.FeatureForT(grd)
                sat_feat_dict_forT, sat_conf_dict_forT = self.FeatureForT(sat)
            else:
                grd_feat_dict_forT, grd_conf_dict_forT = self.GrdFeatureForT(grd)
                sat_feat_dict_forT, sat_conf_dict_forT = self.SatFeatureForT(sat)
            # sat_feat_dict_forT = {}
            # sat_uncer_dict_forT = {}
            for _, level in enumerate(self.level):
                # meter_per_pixel = self.meters_per_pixel[level]
                sat_feat = sat_feat_dict_forT[level]
                grd_feat = grd_feat_dict_forT[level]

                A = sat_feat.shape[-1]

                grd_feat_proj, grd_conf_proj, grd_uv = self.project_grd_to_map(
                    grd_feat, grd_conf_dict_forT[level], pred_rot, shift_u, shift_v, level, meter_per_pixel)

                crop_H = int(A // 2)
                crop_W = int(A // 2)
                g2s_feat = TF.center_crop(grd_feat_proj, [crop_H, crop_W])
                # if self.args.ConfGrd == 0:
                #     g2s_conf = TF.center_crop(grd_conf_proj, [crop_H, crop_W])
                #     g2s_feat = F.normalize((g2s_feat).reshape(B, -1)).reshape(B, -1, crop_H, crop_W)
                # elif self.args.ConfGrd == 1:
                #     g2s_conf = TF.center_crop(grd_conf_proj, [crop_H, crop_W])
                #     g2s_feat = F.normalize((g2s_feat * g2s_conf).reshape(B, -1)).reshape(B, -1, crop_H, crop_W)
                # elif self.args.ConfGrd == 2:
                #     vis_mask = self.VisMask(grd_conf_proj, grd_uv.permute(0, 3, 1, 2), grd_feat_proj)
                #     g2s_conf = TF.center_crop(vis_mask, [crop_H, crop_W])
                #     g2s_feat = F.normalize((g2s_feat * g2s_conf).reshape(B, -1)).reshape(B, -1, crop_H, crop_W)
                #
                # if self.args.ConfSat == 0:
                #     sat_feat_dict_forT[level] = sat_feat
                # else:
                #     sat_feat_dict_forT[level] = F.normalize((sat_feat * sat_conf_dict_forT[level]).reshape(B, -1))\
                #         .reshape(B, *sat_feat.shape[1:])

                if self.args.ConfGrd == 2:
                    vis_mask = self.VisMask(grd_conf_proj, grd_uv.permute(0, 3, 1, 2), grd_feat_proj)
                    g2s_conf = TF.center_crop(vis_mask, [crop_H, crop_W])
                else:
                    g2s_conf = TF.center_crop(grd_conf_proj, [crop_H, crop_W])

                g2s_feat_dict[level] = g2s_feat
                g2s_conf_dict[level] = g2s_conf

            return sat_feat_dict_forT, sat_conf_dict_forT, g2s_feat_dict, g2s_conf_dict, shift_lats, shift_lons, pred_rot

    def forward(self, sat_rt, sat, grd, meter_per_pixel, gt_rot=None, gt_shift_u=None, gt_shift_v=None, stage=None, loop=None, save_dir=None):

        if self.args.task == '2DoF':
            return self.forward2DoF(sat, grd, meter_per_pixel, gt_rot, loop, save_dir)
        else:
            return self.forward3DoF(sat_rt, sat, grd, meter_per_pixel, gt_rot, gt_shift_u, gt_shift_v, stage)



from models_kitti import batch_wise_cross_corr, weak_supervise_loss, corr_for_accurate_translation_supervision


def Weakly_supervised_loss_w_GPS_error(corr_maps, gt_shift_u, gt_shift_v, args, meters_per_pixel, GPS_error=5):
    '''
    corr_maps: dict, key -- level; value -- corr map with shape of [M, N, H, W]
    gt_shift_u: [B]
    gt_shift_v: [B]
    meters_per_pixel: [B], corresponding to original image size
    GPS_error: scalar, in terms of meters
    '''
    matching_losses = []

    # ---------- preparing for GPS error Loss -------
    levels = [int(item) for item in args.level.split('_')]

    GPS_error_losses = []

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
        w = (torch.round(W / 2 - 0.5 + gt_shift_u * 512 / np.power(2, 3 - level) / 4)).long()    # [B]
        h = (torch.round(H / 2 - 0.5 + gt_shift_v * 512 / np.power(2, 3 - level) / 4)).long()    # [B]
        radius = (torch.ceil(GPS_error / (meters_per_pixel * np.power(2, 3 - level)))).long()
        GPS_dis = []
        for b_idx in range(M):
            # GPS_dis.append(torch.min(corr[b_idx, b_idx, h[b_idx]-radius: h[b_idx]+radius, w[b_idx]-radius: w[b_idx]+radius]))
            start_h = torch.max(torch.tensor(0).long(), h[b_idx] - radius[b_idx])
            end_h = torch.min(torch.tensor(corr.shape[2]).long(), h[b_idx] + radius[b_idx])
            start_w = torch.max(torch.tensor(0).long(), w[b_idx] - radius[b_idx])
            end_w = torch.min(torch.tensor(corr.shape[3]).long(), w[b_idx] + radius[b_idx])
            GPS_dis.append(torch.min(
                corr[b_idx, b_idx, start_h: end_h, start_w: end_w]))
        GPS_error_losses.append(torch.abs(torch.stack(GPS_dis) - pos))

    return torch.mean(torch.stack(matching_losses, dim=0)), torch.mean(torch.stack(GPS_error_losses, dim=0))



def GT_triplet_loss(corr_maps, gt_shift_u, gt_shift_v, args):
    '''
    Used when GT GPS lables are highly reliable.
    This function does not handle the rotation issue.
    '''
    levels = [int(item) for item in args.level.split('_')]

    losses = []
    # for level in range(len(corr_maps)):
    for _, level in enumerate(levels):
        corr = corr_maps[level]
        B, corr_H, corr_W = corr.shape

        w = torch.round(corr_W / 2 - 0.5 + gt_shift_u * 512 / np.power(2, 3 - level) / 4)
        h = torch.round(corr_H / 2 - 0.5 + gt_shift_v * 512 / np.power(2, 3 - level) / 4)

        # import pdb; pdb.set_trace()
        pos = corr[range(B), h.long(), w.long()]  # [B]
        pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))

        losses.append(loss)

    return torch.mean(torch.stack(losses, dim=0))



def corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args, sat_uncer_dict=None):
    '''
    to be used during inference
    '''

    level = max([int(item) for item in args.level.split('_')])

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

    if args.use_uncertainty:
        denominator = denominator * TF.center_crop(sat_uncer_dict[level], [corr.shape[1], corr.shape[2]])[:, 0]

    denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

    corr = corr / denominator

    B, corr_H, corr_W = corr.shape

    max_index = torch.argmax(corr.reshape(B, -1), dim=1)
    pred_u = (max_index % corr_W - corr_W / 2)
    pred_v = (max_index // corr_W - corr_H / 2)

    # if level == 3:
    #     return pred_u, pred_v, corr
    #
    # elif level == 2:
    #     return pred_u * 2, pred_v * 2, corr

    return pred_u * np.power(2, 3 - level), pred_v * np.power(2, 3 - level), corr







