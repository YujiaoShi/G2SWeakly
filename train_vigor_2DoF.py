#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# from logging import _Level
import os

import torchvision.utils

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataLoader.Vigor_dataset import load_vigor_data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.io as scio

import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights

from model_vigor import ModelVIGOR, batch_wise_cross_corr, corr_for_translation, Weakly_supervised_loss_w_GPS_error, \
    corr_for_accurate_translation_supervision, GT_triplet_loss


import numpy as np
import os
import argparse
import time
import matplotlib.pyplot as plt
import cv2

from train_kitti_3DoF import show_cam_on_image


def test(net_test, args, save_path):
    ### net evaluation state
    net_test.eval()

    # dataloader = load_vigor_data(args.batch_size, area=args.area)
    dataloader = load_vigor_data(args.batch_size, area=args.area, rotation_range=args.rotation_range,
                                 train=False, weak_supervise=args.Supervision=='Weakly')

    pred_us = []
    pred_vs = []

    gt_us = []
    gt_vs = []

    start_time = time.time()
    with torch.no_grad():

        for i, Data in enumerate(dataloader, 0):

            grd, sat, gt_shift_u, gt_shift_v, gt_rot, meter_per_pixel = [item.to(device) for item in Data]

            sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, sat_uncer_dict = \
                net(None, sat, grd, meter_per_pixel, gt_rot, gt_shift_u, gt_shift_v, stage=args.stage)

            pred_u, pred_v, corr = corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict,
                                                        args, sat_uncer_dict)

            pred_u = pred_u * meter_per_pixel
            pred_v = pred_v * meter_per_pixel

            pred_us.append(pred_u.data.cpu().numpy())
            pred_vs.append(pred_v.data.cpu().numpy())

            gt_shift_u = gt_shift_u * meter_per_pixel * 512 / 4
            gt_shift_v = gt_shift_v * meter_per_pixel * 512 / 4

            gt_us.append(gt_shift_u.data.cpu().numpy())
            gt_vs.append(gt_shift_v.data.cpu().numpy())

            if i % 20 == 0:
                print(i)

    end_time = time.time()
    duration = (end_time - start_time) / len(dataloader) / args.batch_size

    pred_us = np.concatenate(pred_us, axis=0)
    pred_vs = np.concatenate(pred_vs, axis=0)

    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)

    scio.savemat(os.path.join(save_path, 'result.mat'), {'gt_us': gt_us, 'gt_vs': gt_vs,
                                                         'pred_us': pred_us, 'pred_vs': pred_vs,
                                                         })

    distance = np.sqrt((pred_us - gt_us) ** 2 + (pred_vs - gt_vs) ** 2)  # [N]
    init_dis = np.sqrt(gt_us ** 2 + gt_vs ** 2)


    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'results.txt'), 'a')
    # f.write('====================================\n')
    # f.write('       EPOCH: ' + str(epoch) + '\n')
    # print('====================================')
    # print('       EPOCH: ' + str(epoch))
    line = 'Time per image (second): ' + str(duration) + '\n'
    print(line)
    f.write(line)

    line = 'Distance average: (init, pred)' + str(np.mean(init_dis)) + ' ' + str(np.mean(distance))
    print(line)
    f.write(line + '\n')
    line = 'Distance median: (init, pred)' + str(np.median(init_dis)) + ' ' + str(np.median(distance))
    print(line)
    f.write(line + '\n')


    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()
    # result = np.mean(distance)

    net_test.train()


def val(dataloader, net, args, save_path, epoch, best=0.0, stage=None):
    time_start = time.time()

    net.eval()
    print('batch_size:', args.batch_size, '\n num of batches:', len(dataloader))

    pred_us = []
    pred_vs = []

    gt_us = []
    gt_vs = []

    start_time = time.time()
    with torch.no_grad():

        for i, Data in enumerate(dataloader, 0):

            grd, sat, gt_shift_u, gt_shift_v, gt_rot, meter_per_pixel = [item.to(device) for item in Data]

            sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, sat_uncer_dict = \
                net(None, sat, grd, meter_per_pixel, gt_rot, gt_shift_u, gt_shift_v, stage=args.stage)

            pred_u, pred_v, corr = corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict,
                                                        args, sat_uncer_dict)

            pred_u = pred_u * meter_per_pixel
            pred_v = pred_v * meter_per_pixel

            pred_us.append(pred_u.data.cpu().numpy())
            pred_vs.append(pred_v.data.cpu().numpy())

            gt_shift_u = gt_shift_u * meter_per_pixel * 512 / 4
            gt_shift_v = gt_shift_v * meter_per_pixel * 512 / 4

            gt_us.append(gt_shift_u.data.cpu().numpy())
            gt_vs.append(gt_shift_v.data.cpu().numpy())

            if i % 20 == 0:
                print(i)

    end_time = time.time()
    duration = (end_time - start_time) / len(dataloader) / args.batch_size

    pred_us = np.concatenate(pred_us, axis=0)
    pred_vs = np.concatenate(pred_vs, axis=0)

    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)

    distance = np.sqrt((pred_us - gt_us) ** 2 + (pred_vs - gt_vs) ** 2)  # [N]
    init_dis = np.sqrt(gt_us ** 2 + gt_vs ** 2)


    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'val_results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))

    line = 'args.stage: ' + str(args.stage) + 'stage: ' + str(stage) + '\n'
    print(line)
    f.write(line)

    line = 'Time per image (second): ' + str(duration) + '\n'
    print(line)
    f.write(line)

    line = 'Distance average: (init, pred)' + str(np.mean(init_dis)) + ' ' + str(np.mean(distance))
    print(line)
    f.write(line + '\n')
    line = 'Distance median: (init, pred)' + str(np.median(init_dis)) + ' ' + str(np.median(distance))
    print(line)
    f.write(line + '\n')


    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')


    print('====================================')
    f.write('====================================\n')
    f.close()

    result = np.mean(distance)

    net.train()

    ### save the best params
    if args.stage > 0 or (args.stage == -1 and stage == 2):
        if (result < best):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(net.state_dict(), os.path.join(save_path, 'Model_best.pth'))

    print('Finished Val')
    return result


def train(net, args, save_path):
    bestResult = 0.0

    time_start = time.time()
    for epoch in range(args.resume, args.epochs):
        net.train()

        if args.stage == -1 or args.stage == -2:
            params = list(net.SatFeatureNet.parameters()) + list(net.TransRefine.parameters()) + \
                     list(net.GrdFeatureNet.parameters())
            optimizer = optim.Adam(params, lr=1e-4)

        elif args.stage == 0:
            if args.rotation_range == 0:
                params = net.SatFeatureNet.parameters()
            else:
                params = list(net.SatFeatureNet.parameters()) + list(net.TransRefine.parameters())
            optimizer = optim.Adam(params, lr=1e-4)

        elif args.stage == 1:
            params = net.GrdFeatureNet.parameters()
            optimizer = optim.Adam(params, lr=1e-4)

        elif args.stage == 2:

            if args.share:
                params = list(net.FeatureForT.parameters())  # + list(net.GrdFeatureNet.parameters())
            else:
                params = list(net.GrdFeatureForT.parameters()) + list(net.SatFeatureForT.parameters())  # \
                # + list(net.GrdFeatureNet.parameters())

            optimizer = optim.Adam(params, lr=1e-4)

        else:
            optimizer = optim.Adam(net.parameters(), lr=1e-4)

        trainloader, valloader = load_vigor_data(args.batch_size, area=args.area, rotation_range=args.rotation_range,
                                                 train=True, weak_supervise=args.Supervision=='Weakly', amount=args.amount)

        print('batch_size:', args.batch_size, '\n num of batches:', len(trainloader))

        for Loop, Data in enumerate(trainloader, 0):


            grd, sat, gt_shift_u, gt_shift_v, gt_rot, meter_per_pixel = [item.to(device) for item in Data]

            sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, sat_uncer_dict = \
                net(None, sat, grd, meter_per_pixel, gt_rot, gt_shift_u, gt_shift_v, stage=args.stage, loop=Loop, save_dir=save_path)


            if args.Supervision == 'Fully':
                corr_maps = corr_for_accurate_translation_supervision(sat_feat_dict, sat_conf_dict,
                                                                      g2s_feat_dict, g2s_conf_dict, args, sat_uncer_dict)

                GT_loss = GT_triplet_loss(corr_maps, gt_shift_u, gt_shift_v, args)
                loss = GT_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # This step is responsible for updating weights

                if Loop % 10 == 9:  #
                    time_end = time.time()

                    print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) +
                          ' GT_loss: ' + str(np.round(GT_loss.item(), decimals=2)) +
                          # ' Weak_loss: ' + str(np.round(Weak_loss.item(), decimals=2)) +
                          ' Time: ' + str(time_end - time_start))

                    time_start = time_end

            else:

                corr_maps = batch_wise_cross_corr(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args, sat_uncer_dict)

                if args.visualize:

                    visualize_dir = os.path.join(save_path, 'visualization')
                    if not os.path.exists(visualize_dir):
                        os.makedirs(visualize_dir)

                    level = max([int(item) for item in args.level.split('_')])
                    corr = corr_maps[level]
                    corr_H, corr_W = corr.shape[2:]

                    gt_u1 = gt_shift_u.data.cpu().numpy() * 512/4
                    gt_v1 = gt_shift_v.data.cpu().numpy() * 512/4

                    for g_idx in range(corr.shape[0]):
                        for s_idx in range(corr.shape[1]):
                            max_index = torch.argmin(corr[g_idx, s_idx].reshape(-1)).data.cpu().numpy()

                            pred_u = (max_index % corr_W - corr_W / 2) * np.power(2, 3 - level)
                            pred_v = (max_index // corr_W - corr_H / 2) * np.power(2, 3 - level)

                            prob_map = cv2.resize(corr[g_idx, s_idx].data.cpu().numpy(),
                                                  (corr.shape[3] * 2, corr.shape[2] * 2))  # [25:285, 25:285]
                            img = sat[s_idx].permute(1, 2, 0).data.cpu().numpy()[
                                  256 - prob_map.shape[0] // 2: -256 + prob_map.shape[0] // 2,
                                  256 - prob_map.shape[0] // 2:-256 + prob_map.shape[0] // 2, :]

                            overlay = show_cam_on_image(img, prob_map, False, cv2.COLORMAP_HSV)

                            fig, ax = plt.subplots()
                            shw = ax.imshow(overlay)
                            A = overlay.shape[0]
                            # init = ax.scatter(A / 2, A / 2, color='r', linewidth=1, edgecolor="w", s=160, zorder=2)
                            pred = ax.scatter(pred_u + A / 2, pred_v + A / 2, linewidth=1, edgecolor="w", color='r',
                                              s=240, zorder=2)

                            ax.axis('off')

                            # import pdb; pdb.set_trace()

                            if g_idx == s_idx:

                                gt = ax.scatter(gt_u1[g_idx] + A / 2, gt_v1[g_idx] + A / 2, color='g', linewidth=1,
                                                edgecolor="w", marker="*",
                                                s=400,
                                                zorder=2)

                                ax.legend([pred, gt], ['Pred', 'GT'], markerscale=1.2, frameon=False, fontsize=16,
                                          edgecolor="w", labelcolor='w', shadow=True, facecolor='b', loc='upper right')


                                plt.savefig(
                                    os.path.join(visualize_dir,
                                                 'pos_' + str(Loop * args.batch_size + g_idx) + '_' + str(
                                                     s_idx) + '.png'),
                                    transparent=True, dpi=150, bbox_inches='tight', pad_inches=-0.1)
                                plt.close()

                            else:
                                ax.legend([pred], ['Pred'], markerscale=1.2, frameon=False,
                                          fontsize=16,
                                          edgecolor="w", labelcolor='w', shadow=True, facecolor='b', loc='upper right')

                                plt.savefig(
                                    os.path.join(visualize_dir,
                                                 'neg_' + str(Loop * args.batch_size + g_idx) + '_' + str(
                                                     s_idx) + '.png'),
                                    transparent=True, dpi=150, bbox_inches='tight', pad_inches=-0.1)

                                plt.close()
                        print('done')


                corr_loss, GPS_loss = Weakly_supervised_loss_w_GPS_error(corr_maps, gt_shift_u, gt_shift_v,
                                                                         args,
                                                                         meter_per_pixel,
                                                                         args.GPS_error)

                loss = corr_loss + args.GPS_error_coe * GPS_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if Loop % 10 == 9:  #
                    time_end = time.time()

                    print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) +
                          ' triplet loss: ' + str(np.round(corr_loss.item(), decimals=4)) +
                          ' GPS loss: ' + str(np.round(GPS_loss.item(), decimals=4)) +
                          ' Time: ' + str(time_end - time_start))

                    time_start = time_end

        print('Save Model ...')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(net.state_dict(), os.path.join(save_path, 'model_' + str(epoch) + '.pth'))

        bestResult = val(valloader, net, args, save_path, epoch, best=bestResult, stage=args.stage)


    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')
    parser.add_argument('--debug', type=int, default=0, help='debug to dump middle processing images')

    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')

    parser.add_argument('--rotation_range', type=float, default=0., help='degree')

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    parser.add_argument('--level', type=str, default='0_2', help=' ')
    parser.add_argument('--channels', type=str, default='64_16_4', help=' ')
    parser.add_argument('--N_iters', type=int, default=1, help='any integer')

    parser.add_argument('--Optimizer', type=str, default='TransV1', help='LM or SGD')
    parser.add_argument('--proj', type=str, default='geo', help='geo, polar, nn, CrossAttn')
    parser.add_argument('--use_uncertainty', type=int, default=0, help='0 or 1')

    parser.add_argument('--area', type=str, default='cross', help='same or cross')
    parser.add_argument('--multi_gpu', type=int, default=0, help='0 or 1')

    parser.add_argument('--ConfGrd', type=int, default=1, help='use confidence or not for grd image')
    parser.add_argument('--ConfSat', type=int, default=0, help='use confidence or not for sat image')

    parser.add_argument('--share', type=int, default=0, help='share feature extractor for grd and sat or not '
                                                             'in translation estimation')

    parser.add_argument('--GPS_error', type=int, default=5, help='')
    parser.add_argument('--GPS_error_coe', type=float, default=0., help='')

    parser.add_argument('--stage', type=int, default=3,
                        help='fix to 3, this is for dataloader')
    parser.add_argument('--task', type=str, default='2DoF',
                        help='')

    parser.add_argument('--Supervision', type=str, default='Weakly',
                        help='Weakly or Fully')

    parser.add_argument('--visualize', type=int, default=0, help='0 or 1')

    parser.add_argument('--sat', type=float, default=0., help='')
    parser.add_argument('--grd', type=float, default=0., help='')
    parser.add_argument('--sat_grd', type=float, default=1., help='')
    parser.add_argument('--amount', type=float, default=1., help='')

    args = parser.parse_args()

    return args


def getSavePath(args):
    restore_path = './ModelsVIGOR/' + str(args.task) \
                   + '/' + args.area + '_rot' + str(args.rotation_range) \
                   + '_' + str(args.proj) \
                   + '_Level' + args.level + '_Channels' + args.channels

    save_path = restore_path

    if args.ConfGrd:
        save_path = save_path + '_ConfGrd'
    if args.ConfSat:
        save_path = save_path + '_ConfSat'


    if args.GPS_error_coe > 0:
        save_path = save_path + '_GPSerror' + str(args.GPS_error) + '_Coe' + str(args.GPS_error_coe)

    if args.share:
        save_path = save_path + '_Share'

    save_path += '_' + args.Supervision

    if args.amount < 1:
        save_path = save_path + str(args.amount) + '%'

    print('save_path:', save_path)

    return save_path, restore_path


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    np.random.seed(2022)

    args = parse_args()

    save_path, restore_path = getSavePath(args)

    net = ModelVIGOR(args, device)
    if args.multi_gpu:
        net = nn.DataParallel(net, dim=0)

    net.to(device)

    if args.test:
        net.load_state_dict(torch.load(os.path.join(save_path, 'Model_best.pth')), strict=False)
        current = test(net, args, save_path)

    else:

        if args.resume:
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(args.resume - 1) + '.pth')),
                                strict=False)
            print("resume from " + 'model_' + str(args.resume - 1) + '.pth')

        if args.visualize:
            net.load_state_dict(torch.load(os.path.join(save_path, 'Model_best.pth')), strict=False)

        train(net, args, save_path)

