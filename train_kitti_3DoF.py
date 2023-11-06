import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from dataLoader.KITTI_dataset import load_train_data, load_test1_data, load_test2_data
import scipy.io as scio

import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights

# from models_ford import loss_func, loss_func_l2
from models_kitti import Model, batch_wise_cross_corr, corr_for_translation, weak_supervise_loss, \
    Weakly_supervised_loss_w_GPS_error, corr_for_accurate_translation_supervision, GT_triplet_loss, loss_func

import numpy as np
import os
import argparse
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = True,
                      colormap: int = cv2.COLORMAP_RAINBOW) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    mask = np.uint8(255 - 255 * (mask/mask.max()))
    H, W = mask.shape
    heatmap = cv2.applyColorMap(mask, colormap)
    # heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # heatmap = np.float32(heatmap) / 255
    heatmap = np.float32(heatmap)/(255*2)

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    # cam = 0.3*heatmap + 0.7*img
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def test1(net_test, args, save_path, epoch):

    net_test.eval()

    dataloader = load_test1_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    
    pred_lons = []
    pred_lats = []
    pred_oriens = []

    pred_lons_neuralOpt = []
    pred_lats_neuralOpt = []
    
    gt_lons = []
    gt_lats = []
    gt_oriens = []

    start_time = time.time()

    with torch.no_grad():
        for i, Data in enumerate(dataloader, 0):
            sat_align_cam, sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device) for
                                                                                                        item in Data[:7]]

            sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, shift_lats, shift_lons, thetas = \
                net(sat_align_cam, sat_map, grd_left_imgs, left_camera_k, gt_heading)

            pred_u, pred_v, corr = corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict,
                                                        args,
                                                        net_test.meters_per_pixel,
                                                        gt_heading=gt_heading)
            # gt heading here is just to decompose the pred_u & pred_v in the lateral and longitudinal direction
            # for evaluation purpose only

            pred_orien = thetas[:, -1, -1]

            if args.visualize:

                visualize_dir = os.path.join(save_path, 'visualization')
                if not os.path.exists(visualize_dir):
                    os.makedirs(visualize_dir)

                pred_u1 = 512 / 2 - 0.5 + pred_u.data.cpu().numpy()
                pred_v1 = 512 / 2 - 0.5 + pred_v.data.cpu().numpy()
                pred_angle = pred_orien.data.cpu().numpy() * args.rotation_range / 180 * np.pi

                cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
                sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
                gt_delta_x = - gt_shift_u[:, 0] * args.shift_range_lon
                gt_delta_y = - gt_shift_v[:, 0] * args.shift_range_lat
                gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
                gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos
                gt_u1 = torch.round(512 / 2 - 0.5 + gt_delta_x_rot / net_test.meters_per_pixel[3]).data.cpu().numpy()
                gt_v1 = torch.round(512 / 2 - 0.5 + gt_delta_y_rot / net_test.meters_per_pixel[3]).data.cpu().numpy()
                gt_angle = gt_heading[:, 0].data.cpu().numpy() * args.rotation_range / 180 * np.pi

                for b_idx in range(sat_map.shape[0]):
                    img = sat_map[b_idx].permute(1, 2, 0).data.cpu().numpy()[126: -126, 126:-126, :]
                    prob_map = np.asarray(Image.fromarray(corr[b_idx].data.cpu().numpy()).resize((corr.shape[2]*2, corr.shape[1]*2)))[25:285, 25:285]
                    overlay = show_cam_on_image(img, prob_map, False, cv2.COLORMAP_HSV)

                    fig, ax = plt.subplots()
                    ax.imshow(overlay)
                    idx = 0
                    A = 512
                    init = ax.scatter(A / 2, A / 2, color='r', linewidth=1, edgecolor="w", s=60, zorder=2)
                    pred = ax.scatter(pred_u1[idx], pred_v1[idx], linewidth=1, edgecolor="w", color='y', marker="^",
                                      s=60, zorder=2)
                    gt = ax.scatter(gt_u1[idx], gt_v1[idx], color='g', linewidth=1, edgecolor="w", marker="*", s=110,
                                    zorder=2)

                    ax.legend((init, pred, gt), ('Init', 'Pred', 'GT'), markerscale=1.2, frameon=False, fontsize=10,
                              edgecolor="w", labelcolor='w', shadow=True, facecolor='b', loc=(0.548, 0.589))
                    ax.quiver(A / 2, A / 2, 1, 1, angles=0, color='r', zorder=2)
                    ax.quiver(pred_u1[idx], pred_v1[idx],
                              np.cos(pred_angle),
                              np.sin(pred_angle),
                              color='y', zorder=2)
                    ax.quiver(gt_u1[idx], gt_v1[idx],
                              np.cos(gt_angle),
                              np.sin(gt_angle),
                              color='g', zorder=2)
                    ax.axis('off')
                    plt.savefig(os.path.join(visualize_dir, 'pose_' + str( i*args.batch_size + b_idx) + '.png'),
                                transparent=True, dpi=150, bbox_inches='tight', pad_inches=-0.1)
                    plt.close()
                    print('done')


            pred_lons.append(pred_u.data.cpu().numpy())
            pred_lats.append(pred_v.data.cpu().numpy())
            pred_oriens.append(pred_orien.data.cpu().numpy() * args.rotation_range)

            pred_lons_neuralOpt.append(shift_lons[:, -1, -1].data.cpu().numpy())
            pred_lats_neuralOpt.append(shift_lats[:, -1, -1].data.cpu().numpy())

            gt_lons.append(gt_shift_u[:, 0].data.cpu().numpy() * args.shift_range_lon)
            gt_lats.append(gt_shift_v[:, 0].data.cpu().numpy() * args.shift_range_lat)
            gt_oriens.append(gt_heading[:, 0].data.cpu().numpy() * args.rotation_range)


            if i % 20 == 0:
                print(i)
    end_time = time.time()
    duration = (end_time - start_time) / len(dataloader) / args.batch_size

    pred_lons = np.concatenate(pred_lons, axis=0)
    pred_lats = np.concatenate(pred_lats, axis=0)
    pred_oriens = np.concatenate(pred_oriens, axis=0)

    pred_lons_neuralOpt = np.concatenate(pred_lons_neuralOpt, axis=0)
    pred_lats_neuralOpt = np.concatenate(pred_lats_neuralOpt, axis=0)
    
    gt_lons = np.concatenate(gt_lons, axis=0)
    gt_lats = np.concatenate(gt_lats, axis=0)
    gt_oriens = np.concatenate(gt_oriens, axis=0)

    scio.savemat(os.path.join(save_path, 'test1_result.mat'), {'gt_lons': gt_lons, 'gt_lats': gt_lats, 'gt_oriens': gt_oriens,
                                                         'pred_lats': pred_lats, 'pred_lons': pred_lons, 'pred_oriens': pred_oriens})

    distance = np.sqrt((pred_lons - gt_lons) ** 2 + (pred_lats - gt_lats) ** 2)  # [N]
    distanc_neuralOpt = np.sqrt((pred_lons_neuralOpt - gt_lons) ** 2 + (pred_lats_neuralOpt - gt_lats) ** 2)  # [N]

    init_dis = np.sqrt(gt_lats ** 2 + gt_lons ** 2)
    
    diff_lats = np.abs(pred_lats - gt_lats)
    diff_lons = np.abs(pred_lons - gt_lons)

    diff_lats_neuralOpt = np.abs(pred_lats_neuralOpt - gt_lats)
    diff_lons_neuralOpt = np.abs(pred_lons_neuralOpt - gt_lons)
   
    angle_diff = np.remainder(np.abs(pred_oriens - gt_oriens), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]

    init_angle = np.abs(gt_oriens)

    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'test1_results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    line = 'Time per image (second): ' + str(duration) + '\n'
    print(line)
    f.write(line)
    line = 'Test1 results:'
    print(line)
    f.write(line + '\n')

    line = 'Distance average: (init, pred by corr, pred by neuralOpt)' + str(np.mean(init_dis)) + ' ' + str(np.mean(distance)) + ' ' + str(np.mean(distanc_neuralOpt))
    print(line)
    f.write(line + '\n')
    line = 'Distance median: (init, pred by corr, pred by neuralOpt)' + str(np.median(init_dis)) + ' ' + str(np.median(distance)) + ' ' + str(np.median(distanc_neuralOpt))
    print(line)
    f.write(line + '\n')

    line = 'Lateral average: (init, pred by corr, pred by neuralOpt)' + str(np.mean(np.abs(gt_lats))) + ' ' + str(np.mean(diff_lats)) + ' ' + str(np.mean(diff_lats_neuralOpt))
    print(line)
    f.write(line + '\n')
    line = 'Lateral median: (init, pred by corr, pred by neuralOpt)' + str(np.median(np.abs(gt_lats))) + ' ' + str(np.median(diff_lats)) + ' ' + str(np.median(diff_lats_neuralOpt))
    print(line)
    f.write(line + '\n')

    line = 'Longitudinal average: (init by corr, pred, pred by neuralOpt)' + str(np.mean(np.abs(gt_lons))) + ' ' + str(np.mean(diff_lons)) + ' ' + str(np.mean(diff_lons_neuralOpt))
    print(line)
    f.write(line + '\n')
    line = 'Longitudinal median: (init by corr, pred, pred by neuralOpt)' + str(np.median(np.abs(gt_lons))) + ' ' + str(np.median(diff_lons)) + ' ' + str(np.median(diff_lons_neuralOpt))
    print(line)
    f.write(line + '\n')

    line = 'Angle average (init, pred): ' + str(np.mean(np.abs(gt_oriens))) + ' ' + str(np.mean(angle_diff))
    print(line)
    f.write(line + '\n')
    line = 'Angle median (init, pred): ' + str(np.median(np.abs(gt_oriens))) + ' ' + str(np.median(angle_diff))
    print(line)
    f.write(line + '\n')

    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100
        pred_opt = np.sum(distanc_neuralOpt < metrics[idx]) / distanc_neuralOpt.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (init, pred by corr, pred by neuralOpt): ' + str(init) + ' ' + str(pred) + ' ' + str(pred_opt)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(metrics)):
        pred = np.sum(diff_lats < metrics[idx]) / diff_lats.shape[0] * 100
        init = np.sum(np.abs(gt_lats) < metrics[idx]) / gt_lats.shape[0] * 100
        pred_opt = np.sum(diff_lats_neuralOpt < metrics[idx]) / diff_lats_neuralOpt.shape[0] * 100

        line = 'lateral within ' + str(metrics[idx]) + ' meters (init, pred by corr, pred by neuralOpt): ' + str(init) + ' ' + str(pred) + ' ' + str(pred_opt)
        print(line)
        f.write(line + '\n')

    for idx in range(len(metrics)):

        pred = np.sum(diff_lons < metrics[idx]) / diff_lons.shape[0] * 100
        init = np.sum(np.abs(gt_lons) < metrics[idx]) / gt_lons.shape[0] * 100
        pred_opt = np.sum(diff_lons_neuralOpt < metrics[idx]) / diff_lons_neuralOpt.shape[0] * 100

        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (init, pred by corr, pred by neuralOpt): ' + str(init) + ' ' + str(pred) + ' ' + str(pred_opt)
        print(line)
        f.write(line + '\n')

    
    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (init, pred by corr, pred by neuralOpt): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')
    f.close()

    net_test.train()

    return


def test2(net_test, args, save_path):

    net_test.eval()

    dataloader = load_test2_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    
    pred_lons = []
    pred_lats = []
    pred_oriens = []

    pred_lons_neuralOpt = []
    pred_lats_neuralOpt = []

    gt_lons = []
    gt_lats = []
    gt_oriens = []

    with torch.no_grad():
        for i, Data in enumerate(dataloader, 0):
            sat_align_cam, sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device)
                                                                                                        for
                                                                                                        item in
                                                                                                        Data[:7]]
            # if args.stage == 0:
            sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, shift_lats, shift_lons, thetas = \
                net(sat_align_cam, sat_map, grd_left_imgs, left_camera_k, gt_heading)
            pred_orien = thetas[:, -1, -1]
            # else:
            #     sat_feat_dict, sat_uncer_dict, g2s_feat_dict, g2s_conf_dict, shift_lats, shift_lons, pred_orien = \
            #         net(sat_align_cam, sat_map, grd_left_imgs, left_camera_k, gt_heading)
            #     pred_orien = pred_orien[:, 0]

            pred_u, pred_v, corr = corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict,
                                                        args,
                                                        net_test.meters_per_pixel,
                                                        gt_heading=gt_heading)
            # gt heading here is just to decompose the pred_u & pred_v in the lateral and longitudinal direction
            # for evaluation purpose only

            # pred_orien = thetas[:, -1, -1]
            pred_lons.append(pred_u.data.cpu().numpy())
            pred_lats.append(pred_v.data.cpu().numpy())
            pred_oriens.append(pred_orien.data.cpu().numpy() * args.rotation_range)

            pred_lons_neuralOpt.append(shift_lons[:, -1, -1].data.cpu().numpy())
            pred_lats_neuralOpt.append(shift_lats[:, -1, -1].data.cpu().numpy())

            gt_lons.append(gt_shift_u[:, 0].data.cpu().numpy() * args.shift_range_lon)
            gt_lats.append(gt_shift_v[:, 0].data.cpu().numpy() * args.shift_range_lat)
            gt_oriens.append(gt_heading[:, 0].data.cpu().numpy() * args.rotation_range)

            if i % 20 == 0:
                print(i)

    pred_lons = np.concatenate(pred_lons, axis=0)
    pred_lats = np.concatenate(pred_lats, axis=0)
    pred_oriens = np.concatenate(pred_oriens, axis=0)

    pred_lons_neuralOpt = np.concatenate(pred_lons_neuralOpt, axis=0)
    pred_lats_neuralOpt = np.concatenate(pred_lats_neuralOpt, axis=0)

    gt_lons = np.concatenate(gt_lons, axis=0)
    gt_lats = np.concatenate(gt_lats, axis=0)
    gt_oriens = np.concatenate(gt_oriens, axis=0)

    scio.savemat(os.path.join(save_path, 'result.mat'), {'gt_lons': gt_lons, 'gt_lats': gt_lats, 'gt_oriens': gt_oriens, 
                                                         'pred_lats': pred_lats, 'pred_lons': pred_lons, 'pred_oriens': pred_oriens})
    # scio.savemat(os.path.join(save_path, 'result.mat'), {'gt_lons': gt_lons, 'gt_lats': gt_lats, 
    #                                                      'pred_lats': pred_lats, 'pred_lons': pred_lons})

    distance = np.sqrt((pred_lons - gt_lons) ** 2 + (pred_lats - gt_lats) ** 2)  # [N]
    distanc_neuralOpt = np.sqrt((pred_lons_neuralOpt - gt_lons) ** 2 + (pred_lats_neuralOpt - gt_lats) ** 2)  # [N]

    init_dis = np.sqrt(gt_lats ** 2 + gt_lons ** 2)

    diff_lats = np.abs(pred_lats - gt_lats)
    diff_lons = np.abs(pred_lons - gt_lons)

    diff_lats_neuralOpt = np.abs(pred_lats_neuralOpt - gt_lats)
    diff_lons_neuralOpt = np.abs(pred_lons_neuralOpt - gt_lons)
   
    angle_diff = np.remainder(np.abs(pred_oriens - gt_oriens), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]
  
    init_angle = np.abs(gt_oriens)

    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'test2_results.txt'), 'a')
    print('-------------------------')
    f.write('------------------------\n')
    line = 'Test2 results:'
    print(line)
    f.write(line + '\n')
    
    line = 'Distance average: (init, pred by corr, pred by neuralOpt)' + str(np.mean(init_dis)) + ' ' + str(np.mean(distance)) + ' ' + str(np.mean(distanc_neuralOpt))
    print(line)
    f.write(line + '\n')
    line = 'Distance median: (init, pred by corr, pred by neuralOpt)' + str(np.median(init_dis)) + ' ' + str(np.median(distance)) + ' ' + str(np.median(distanc_neuralOpt))
    print(line)
    f.write(line + '\n')

    line = 'Lateral average: (init, pred by corr, pred by neuralOpt)' + str(np.mean(np.abs(gt_lats))) + ' ' + str(np.mean(diff_lats)) + ' ' + str(np.mean(diff_lats_neuralOpt))
    print(line)
    f.write(line + '\n')
    line = 'Lateral median: (init, pred by corr, pred by neuralOpt)' + str(np.median(np.abs(gt_lats))) + ' ' + str(np.median(diff_lats)) + ' ' + str(np.median(diff_lats_neuralOpt))
    print(line)
    f.write(line + '\n')

    line = 'Longitudinal average: (init, pred by corr, pred by neuralOpt)' + str(np.mean(np.abs(gt_lons))) + ' ' + str(np.mean(diff_lons)) + ' ' + str(np.mean(diff_lons_neuralOpt))
    print(line)
    f.write(line + '\n')
    line = 'Longitudinal median: (init, pred by corr, pred by neuralOpt)' + str(np.median(np.abs(gt_lons))) + ' ' + str(np.median(diff_lons)) + ' ' + str(np.median(diff_lons_neuralOpt))
    print(line)
    f.write(line + '\n')

    line = 'Angle average (init, pred): ' + str(np.mean(np.abs(gt_oriens))) + ' ' + str(np.mean(angle_diff))
    print(line)
    f.write(line + '\n')
    line = 'Angle median (init, pred): ' + str(np.median(np.abs(gt_oriens))) + ' ' + str(np.median(angle_diff))
    print(line)
    f.write(line + '\n')

    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100
        pred_opt = np.sum(distanc_neuralOpt < metrics[idx]) / distanc_neuralOpt.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (init, pred by corr, pred by neuralOpt): ' + str(
            init) + ' ' + str(pred) + ' ' + str(pred_opt)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(metrics)):
        pred = np.sum(diff_lats < metrics[idx]) / diff_lats.shape[0] * 100
        init = np.sum(np.abs(gt_lats) < metrics[idx]) / gt_lats.shape[0] * 100
        pred_opt = np.sum(diff_lats_neuralOpt < metrics[idx]) / diff_lats_neuralOpt.shape[0] * 100

        line = 'lateral within ' + str(metrics[idx]) + ' meters (init, pred by corr, pred by neuralOpt): ' + str(
            init) + ' ' + str(pred) + ' ' + str(pred_opt)
        print(line)
        f.write(line + '\n')

    for idx in range(len(metrics)):
        pred = np.sum(diff_lons < metrics[idx]) / diff_lons.shape[0] * 100
        init = np.sum(np.abs(gt_lons) < metrics[idx]) / gt_lons.shape[0] * 100
        pred_opt = np.sum(diff_lons_neuralOpt < metrics[idx]) / diff_lons_neuralOpt.shape[0] * 100

        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (init, pred by corr, pred by neuralOpt): ' + str(
            init) + ' ' + str(pred) + ' ' + str(pred_opt)
        print(line)
        f.write(line + '\n')

    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (init, pred by corr, pred by neuralOpt): ' + str(
            init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()
    result = np.sum((diff_lats < metrics[0])) / diff_lats.shape[0] * 100

    net_test.train()
    return result


def train(net, args, save_path):
    bestRankResult = 0.0

    time_start = time.time()
    for epoch in range(args.resume, args.epochs):
        net.train()

        # optimizer = optim.Adam(net.parameters(), lr=base_lr)
        if args.stage == 0:
            if args.rotation_range == 0:
                params = net.SatFeatureNet.parameters()
            else:
                params = list(net.SatFeatureNet.parameters()) + list(net.TransRefine.parameters())
            optimizer = optim.Adam(params, lr=1e-4)
        elif args.stage == 1:

            if args.share:
                params = net.FeatureForT.parameters()
            else:
                params = list(net.GrdFeatureForT.parameters()) + list(net.SatFeatureForT.parameters())

            optimizer = optim.Adam(params, lr=1e-4)

        trainloader = load_train_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range,
                                      weak_supervise=True, train_noisy=args.train_noisy, stage=args.stage,
                                      data_amount=args.supervise_amount)

        print('batch_size:', args.batch_size, '\n num of batches:', len(trainloader))

        for Loop, Data in enumerate(trainloader, 0):

            sat_align_cam, sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device) for item in Data[:7]]

            sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, shift_lats, shift_lons, thetas = \
                net(sat_align_cam, sat_map, grd_left_imgs, left_camera_k, gt_heading, gt_shift_u, gt_shift_v, loop=Loop, save_dir=save_path)

            if args.stage == 0:
                opt_loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                    shift_lat_last, shift_lon_last, theta_last, \
                    = loss_func(shift_lats, shift_lons, thetas, gt_shift_v[:, 0], gt_shift_u[:, 0], gt_heading[:, 0],
                                torch.exp(-net.coe_R), torch.exp(-net.coe_R), torch.exp(-net.coe_R))


                corr_maps = corr_for_accurate_translation_supervision(sat_feat_dict, sat_conf_dict,
                                                             g2s_feat_dict, g2s_conf_dict, args)

                corr_loss = GT_triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading, args, net.meters_per_pixel)

                if args.rotation_range == 0:
                    loss = corr_loss
                else:
                    loss = opt_loss + \
                           corr_loss * torch.exp(-net.coe_T) + \
                           net.coe_T + net.coe_R

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # This step is responsible for updating weights

                if Loop % 10 == 9:  #
                    time_end = time.time()

                    print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) +
                          ' DeltaR: ' + str(np.round(thetas_decrease.item(), decimals=2)) +
                          ' FinalR: ' + str(np.round(theta_last.item(), decimals=2)) +
                          ' triplet loss: ' + str(np.round(corr_loss.item(), decimals=4)) +
                          ' Time: ' + str(time_end - time_start))

                    time_start = time_end

            elif args.stage > 0:

                corr_maps = batch_wise_cross_corr(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args)
                # corr_loss = weak_supervise_loss(corr_maps)

                if args.visualize:

                    visualize_dir = os.path.join(save_path, 'visualization')
                    if not os.path.exists(visualize_dir):
                        os.makedirs(visualize_dir)

                    level = max([int(item) for item in args.level.split('_')])
                    corr = corr_maps[level]
                    corr_H, corr_W = corr.shape[2:]

                    cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
                    sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
                    gt_delta_x = - gt_shift_u[:, 0] * args.shift_range_lon
                    gt_delta_y = - gt_shift_v[:, 0] * args.shift_range_lat
                    gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
                    gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos
                    gt_u1 = (gt_delta_x_rot / net.meters_per_pixel[3]).data.cpu().numpy()
                    gt_v1 = (gt_delta_y_rot / net.meters_per_pixel[3]).data.cpu().numpy()
                    gt_angle = gt_heading[:, 0].data.cpu().numpy() * args.rotation_range / 180 * np.pi

                    for g_idx in range(corr.shape[0]):
                        for s_idx in range(corr.shape[1]):
                            max_index = torch.argmin(corr[g_idx, s_idx].reshape(-1)).data.cpu().numpy()

                            pred_u = (max_index % corr_W - corr_W / 2) * np.power(2, 3 - level)
                            pred_v = (max_index // corr_W - corr_H / 2) * np.power(2, 3 - level)
                            pred_angle = thetas[g_idx, -1, -1].data.cpu().numpy() * args.rotation_range / 180 * np.pi

                            prob_map = cv2.resize(corr[g_idx, s_idx].data.cpu().numpy(), (corr.shape[3] * 2, corr.shape[2] * 2))#[25:285, 25:285]
                            img = sat_map[s_idx].permute(1, 2, 0).data.cpu().numpy()[256-prob_map.shape[0]//2: -256+prob_map.shape[0]//2, 256-prob_map.shape[0]//2:-256+prob_map.shape[0]//2, :]

                            overlay = show_cam_on_image(img, prob_map, False, cv2.COLORMAP_HSV)

                            fig, ax = plt.subplots()
                            shw = ax.imshow(overlay)

                            norm = colors.Normalize(vmin=0, vmax=1)
                            shw.set_norm(norm)
                            shw.set_cmap('jet')
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes('right', size='5%', pad=0.05)
                            fig.colorbar(shw, cax=cax, orientation='vertical')

                            A = overlay.shape[0]
                            # init = ax.scatter(A / 2, A / 2, color='r', linewidth=1, edgecolor="w", s=160, zorder=2)
                            pred = ax.scatter(pred_u + A / 2, pred_v + A / 2, linewidth=1, edgecolor="w", color='r',
                                              s=240, zorder=2)
                            # , marker = "^"

                            s = 1
                            # ax.quiver(A / 2, A / 2, 1, 1, angles=0, color='r', zorder=2, scale_units='width', scale=15,
                            #           width=0.015, headwidth=3, headlength=4, headaxislength=3.5)
                            ax.quiver(pred_u + A / 2, pred_v + A / 2,
                                      np.cos(pred_angle),
                                      np.sin(pred_angle),
                                      color='r', zorder=2, scale_units='width', scale=12,
                                      width=0.015, headwidth=3, headlength=4, headaxislength=3.5)

                            ax.axis('off')

                            # import pdb; pdb.set_trace()

                            if g_idx == s_idx:
                                # cbar = plt.colorbar(shw, location='left', orientation='vertical', ticks=[overlay.min(), overlay.max()])
                                # cbar.ax.set_yticklabels(['Low', 'High'])

                                gt = ax.scatter(gt_u1[g_idx] + A / 2, gt_v1[g_idx] + A / 2, color='g', linewidth=1, edgecolor="w", marker="*",
                                                s=400,
                                                zorder=2)

                                ax.legend([pred, gt], ['Pred', 'GT'], markerscale=1.2, frameon=False, fontsize=16,
                                          edgecolor="black", labelcolor='black', shadow=True, facecolor='b', loc='upper center', bbox_to_anchor=(0.5, 1.14), ncols=2)  # , loc='upper right'
                                ax.quiver(gt_u1[g_idx] + A / 2, gt_v1[g_idx] + A / 2,
                                          np.cos(gt_angle)[g_idx],
                                          np.sin(gt_angle)[g_idx],
                                          color='g', zorder=2, scale_units='width', scale=13,
                                      width=0.015, headwidth=3, headlength=4, headaxislength=3.5)



                                plt.savefig(
                                    os.path.join(visualize_dir, 'pos_' + str(Loop * args.batch_size + g_idx) + '_' + str(s_idx) + '.png'),
                                    transparent=True, dpi=150, bbox_inches='tight', pad_inches=0) # , bbox_inches='tight'
                                plt.close()

                                break

                            else:
                                ax.legend([pred], ['Pred'], markerscale=1.2, frameon=False,
                                          fontsize=16,
                                          edgecolor="black", labelcolor='black', shadow=True, facecolor='b', loc='upper center', bbox_to_anchor=(0.5, 1.14))

                                # plt.show()
                                plt.savefig(
                                    os.path.join(visualize_dir,
                                                 'neg_' + str(Loop * args.batch_size + g_idx) + '_' + str(
                                                     s_idx) + '.png'),
                                    transparent=True, dpi=150, pad_inches=-0.1)  #, bbox_inches='tight'



                                plt.close()
                        print('done')

                corr_loss, GPS_loss = Weakly_supervised_loss_w_GPS_error(corr_maps, gt_shift_u, gt_shift_v,
                                                                         gt_heading,
                                                                         args,
                                                                         net.meters_per_pixel,
                                                                         args.GPS_error)
                # gt heading here just to compute the GPS position


                loss = args.contrastive_coe * corr_loss + args.GPS_error_coe * GPS_loss

                R_err = torch.abs(thetas[:, -1, -1].reshape(-1) - gt_heading.reshape(-1)).mean() * args.rotation_range

                optimizer.zero_grad()
                # optimizer2.zero_grad()
                loss.backward()
                optimizer.step()
                # optimizer2.step()

                if Loop % 10 == 9:  #
                    time_end = time.time()

                    print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) +
                          ' R error: ' + str(np.round(R_err.item(), decimals=4)) +
                          ' triplet loss: ' + str(np.round(corr_loss.item(), decimals=4)) +
                          ' GPS err loss: ' + str(np.round(GPS_loss.item(), decimals=4)) +
                          ' Time: ' + str(time_end - time_start))

                    time_start = time_end

        print('Save Model ...')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(net.state_dict(), os.path.join(save_path, 'model_' + str(epoch) + '.pth'))

        test1(net, args, save_path, epoch)

        test2(net, args, save_path)

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')

    parser.add_argument('--epochs', type=int, default=3, help='number of training epochs')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  # 1e-2

    parser.add_argument('--rotation_range', type=float, default=10., help='degree')
    parser.add_argument('--shift_range_lat', type=float, default=20., help='meters')
    parser.add_argument('--shift_range_lon', type=float, default=20., help='meters')

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    parser.add_argument('--level', type=str, default='0_2', help=' ')
    parser.add_argument('--channels', type=str, default='64_16_4', help='64_16_4 ')
    parser.add_argument('--N_iters', type=int, default=1, help='any integer')

    # parser.add_argument('--confidence', type=int, default=0, help='use confidence or not')
    parser.add_argument('--ConfGrd', type=int, default=1, help='use confidence or not for grd image')
    parser.add_argument('--ConfSat', type=int, default=0, help='use confidence or not for sat image')

    parser.add_argument('--share', type=int, default=1, help='share feature extractor for grd and sat or not '
                                                             'in translation estimation')

    parser.add_argument('--Optimizer', type=str, default='TransV1', help='LM or SGD')
    parser.add_argument('--proj', type=str, default='geo', help='geo or CrossAttn')

    parser.add_argument('--visualize', type=int, default=0, help='0 or 1')

    parser.add_argument('--multi_gpu', type=int, default=0, help='0 or 1')

    parser.add_argument('--GPS_error', type=int, default=5, help='')
    parser.add_argument('--GPS_error_coe', type=float, default=0., help='')

    parser.add_argument('--stage', type=int, default=1, help='0 or 1, 0 for self-supervised training, 1 for E2E training')
    parser.add_argument('--task', type=str, default='3DoF',
                        help='')

    parser.add_argument('--supervise_amount', type=float, default=1.0,
                        help='0.1, 0.2, 0.3, ..., 1')

    args = parser.parse_args()

    return args


def getSavePath(args):
    save_path= restore_path = './ModelsKitti/3DoF/Stage' + str(args.stage) \
                + '/lat' + str(args.shift_range_lat) + 'm_lon' + str(args.shift_range_lon) + 'm_rot' + str(
        args.rotation_range)  \
                + '_Nit' + str(args.N_iters) + '_' + str(args.Optimizer) + '_' + str(args.proj) \
                + '_Level' + args.level + '_Channels' + args.channels

    if args.ConfGrd and args.stage > 0:
        save_path = save_path + '_ConfGrd'
    if args.ConfSat and args.stage > 0:
        save_path = save_path + '_ConfSat'

    if args.GPS_error_coe > 0 and args.stage > 0:

        save_path = save_path + '_GPSerror' + str(args.GPS_error) + '_Coe' + str(args.GPS_error_coe)


    if args.share and args.stage > 0:
        save_path = save_path + '_Share'

    if args.supervise_amount < 1 and args.stage > 0:
        save_path += '_' + str(args.supervise_amount)


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

    net = Model(args, device=device)

    if args.multi_gpu:
        net = nn.DataParallel(net, dim=0)

    net.to(device)

    if args.test:
        net.load_state_dict(torch.load(os.path.join(
            save_path.replace('lat' + str(args.shift_range_lat) + 'm_lon' + str(args.shift_range_lon), 'lat20.0m_lon20.0'),
            'model_2.pth')), strict=False)
        test1(net, args, save_path, epoch=2)
        test2(net, args, save_path)

    else:

        if args.resume:
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(args.resume - 1) + '.pth')))
            print("resume from " + 'model_' + str(args.resume - 1) + '.pth')
        elif args.stage == 1 and args.rotation_range > 0:

            net.load_state_dict(torch.load(
                os.path.join(restore_path.replace('Stage1', 'Stage0').replace(args.proj, 'geo'), 'model_2.pth')), strict=False)
            print("load pretrained model from Stage0:")
            print(os.path.join(restore_path.replace('Stage1', 'Stage0'),
                               'model_2.pth'))

        if args.visualize:
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_2.pth')), strict=False)
            print('------------------------')
            print("load pretrained model from ", os.path.join(save_path, 'model_2.pth'))
            print('------------------------')

        lr = args.lr

        train(net, args, save_path)



def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
