import os.path

import numpy as np
from PIL import Image
from torchvision import transforms

def features_to_RGB(sat_feat, g2s_feat_center, g2s_conf_center, g2s_feat_gt, g2s_conf_gt, loop, level, save_dir):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    from sklearn.decomposition import PCA

    def reshape_normalize(x):
        '''
        Args:
            x: [B, C, H, W]

        Returns:

        '''
        B, C, H, W = x.shape
        x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator==0, 1, denominator)
        return x / denominator

    def normalize(x):
        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator == 0, 1, denominator)
        return x / denominator

    sat_feat = sat_feat.data.cpu().numpy()  # [B, C, H, W]
    g2s_feat_center = g2s_feat_center.data.cpu().numpy()  # [B, C, H, W]
    g2s_feat_gt = g2s_feat_gt.data.cpu().numpy()

    B, C, A, _ = sat_feat.shape

    flatten = np.concatenate([sat_feat, g2s_feat_center, g2s_feat_gt], axis=0)

    # if level == 0:
    pca = PCA(n_components=3)
    pca.fit(reshape_normalize(flatten))

    sat_feat_new = ((normalize(pca.transform(reshape_normalize(sat_feat))) + 1 )/ 2).reshape(B, A, A, 3)

    mask_center = g2s_conf_center[:, 0, :, :, None].data.cpu().numpy()
    mask_center = mask_center / mask_center.max()
    mask = np.linalg.norm(g2s_feat_center, axis=1)[:, :, :, None] > 0
    g2s_feat_new_center = ((normalize(pca.transform(reshape_normalize(g2s_feat_center))) + 1) / 2).reshape(B, A, A, 3) * mask

    mask_gt = g2s_conf_gt[:, 0, :, :, None].data.cpu().numpy()
    mask_gt = mask_gt / mask_gt.max()
    mask = np.linalg.norm(g2s_feat_gt, axis=1)[:, :, :, None] > 0
    g2s_feat_new_gt = ((normalize(pca.transform(reshape_normalize(g2s_feat_gt))) + 1) / 2).reshape(B, A, A, 3) * mask

    for idx in range(B):
        sat = Image.fromarray((sat_feat_new[idx] * 255).astype(np.uint8))
        sat = sat.resize((512, 512))
        sat.save(os.path.join(save_dir, 'level_' + str(level) + '_sat_feat_' + str(loop * B + idx) + '.png'))

        g2s_center = Image.fromarray((g2s_feat_new_center[idx] * 255).astype(np.uint8))
        g2s_center = g2s_center.resize((512, 512))
        g2s_center.save(os.path.join(save_dir, 'level_' + str(level) + '_g2s_feat_center' + str(loop * B + idx) + '.png'))

        g2s_center = Image.fromarray((g2s_feat_new_center[idx] * mask_center[idx] * 255).astype(np.uint8))
        g2s_center = g2s_center.resize((512, 512))
        g2s_center.save(
            os.path.join(save_dir, 'level_' + str(level) + '_g2s_feat_center_conf' + str(loop * B + idx) + '.png'))

        g2s_gt = Image.fromarray((g2s_feat_new_gt[idx] * 255).astype(np.uint8))
        g2s_gt = g2s_gt.resize((512, 512))
        g2s_gt.save(os.path.join(save_dir, 'level_' + str(level) + '_g2s_feat_gt' + str(loop * B + idx) + '.png'))

        g2s_gt = Image.fromarray((g2s_feat_new_gt[idx] * mask_gt[idx] * 255).astype(np.uint8))
        g2s_gt = g2s_gt.resize((512, 512))
        g2s_gt.save(os.path.join(save_dir, 'level_' + str(level) + '_g2s_feat_gt_conf' + str(loop * B + idx) + '.png'))

    return


def RGB_iterative_pose(sat_img, grd_img, shift_lats, shift_lons, thetas, gt_shift_u, gt_shift_v, gt_theta,
                       meter_per_pixel, args, loop=0, save_dir='./visualize/'):
    '''
    This function is for KITTI dataset
    Args:
        sat_img: [B, C, H, W]
        shift_lats: [B, Niters, Level]
        shift_lons: [B, Niters, Level]
        thetas: [B, Niters, Level]
        meter_per_pixel: scalar

    Returns:

    '''

    import matplotlib.pyplot as plt

    B, _, A, _ = sat_img.shape

    # A = 512 - 128

    shift_lats = (A/2 - shift_lats.data.cpu().numpy() * args.shift_range_lat / meter_per_pixel).reshape([B, -1])
    shift_lons = (A/2 + shift_lons.data.cpu().numpy() * args.shift_range_lon / meter_per_pixel).reshape([B, -1])
    thetas = (- thetas.data.cpu().numpy() * args.rotation_range).reshape([B, -1])

    gt_u = (A/2 + gt_shift_u.data.cpu().numpy() * args.shift_range_lon / meter_per_pixel)
    gt_v = (A/2 - gt_shift_v.data.cpu().numpy() * args.shift_range_lat / meter_per_pixel)
    gt_theta = - gt_theta.cpu().numpy() * args.rotation_range

    for idx in range(B):
        img = np.array(transforms.functional.to_pil_image(sat_img[idx], mode='RGB'))
        # img = img[64:-64, 64:-64]
        # A = img.shape[0]

        fig, ax = plt.subplots()
        ax.imshow(img)
        init = ax.scatter(A/2, A/2, color='r', s=20, zorder=2)
        update = ax.scatter(shift_lons[idx, :-1], shift_lats[idx, :-1], color='m', s=15, zorder=2)
        pred = ax.scatter(shift_lons[idx, -1], shift_lats[idx, -1], color='g', s=20, zorder=2)
        gt = ax.scatter(gt_u[idx], gt_v[idx], color='b', s=20, zorder=2)
        # ax.legend((init, update, pred, gt), ('Init', 'Intermediate', 'Pred', 'GT'),
        #           frameon=False, fontsize=14, labelcolor='r', loc=2)
        # loc=1: upper right
        # loc=3: lower left

        # if args.rotation_range>0:
        init = ax.quiver(A/2, A/2, 1, 1, angles=0, color='r', zorder=2)
        # update = ax.quiver(shift_lons[idx, :], shift_lats[idx, :], 1, 1, angles=thetas[idx, :], color='r')
        pred = ax.quiver(shift_lons[idx, -1], shift_lats[idx, -1], 1, 1, angles=thetas[idx, -1], color='g', zorder=2)
        gt = ax.quiver(gt_u[idx], gt_v[idx], 1, 1, angles=gt_theta[idx], color='b', zorder=2)
        # ax.legend((init, pred, gt), ('pred', 'Updates', 'GT'), frameon=False, fontsize=16, labelcolor='r')
        #
        # # for i in range(shift_lats.shape[1]-1):
        # #     ax.quiver(shift_lons[idx, i], shift_lats[idx, i], shift_lons[idx, i+1], shift_lats[idx, i+1], angles='xy',
        # #               color='r')
        #
        ax.axis('off')

        plt.savefig(os.path.join(save_dir, 'points_' + str(loop * B + idx) + '.png'),
                    transparent=True, dpi=A, bbox_inches='tight')
        plt.close()

        grd = transforms.functional.to_pil_image(grd_img[idx], mode='RGB')
        grd.save(os.path.join(save_dir, 'grd_' + str(loop * B + idx) + '.png'))

        sat = transforms.functional.to_pil_image(sat_img[idx], mode='RGB')
        sat.save(os.path.join(save_dir, 'sat_' + str(loop * B + idx) + '.png'))


def RGB_iterative_pose_ford(sat_img, grd_img, shift_lats, shift_lons, thetas, gt_shift_u, gt_shift_v, gt_theta,
                       meter_per_pixel, args, loop=0, save_dir='./visualize/'):
    '''
    This function is for KITTI dataset
    Args:
        sat_img: [B, C, H, W]
        shift_lats: [B, Niters, Level]
        shift_lons: [B, Niters, Level]
        thetas: [B, Niters, Level]
        meter_per_pixel: scalar

    Returns:

    '''

    import matplotlib.pyplot as plt

    B, _, A, _ = sat_img.shape

    # A = 512 - 128

    shift_lats = (A/2 - shift_lats.data.cpu().numpy() * args.shift_range_lat / meter_per_pixel).reshape([B, -1])
    shift_lons = (A/2 - shift_lons.data.cpu().numpy() * args.shift_range_lon / meter_per_pixel).reshape([B, -1])
    thetas = (- thetas.data.cpu().numpy() * args.rotation_range).reshape([B, -1])

    gt_u = (A/2 - gt_shift_u.data.cpu().numpy() * args.shift_range_lat / meter_per_pixel)
    gt_v = (A/2 - gt_shift_v.data.cpu().numpy() * args.shift_range_lon / meter_per_pixel)
    gt_theta = - gt_theta.cpu().numpy() * args.rotation_range

    for idx in range(B):
        img = np.array(transforms.functional.to_pil_image(sat_img[idx], mode='RGB'))
        # img = img[64:-64, 64:-64]
        # A = img.shape[0]

        fig, ax = plt.subplots()
        ax.imshow(img)
        init = ax.scatter(A/2, A/2, color='r', s=20, zorder=2)
        update = ax.scatter(shift_lats[idx, :-1], shift_lons[idx, :-1], color='m', s=15, zorder=2)
        pred = ax.scatter(shift_lats[idx, -1], shift_lons[idx, -1], color='g', s=20, zorder=2)
        gt = ax.scatter(gt_u[idx], gt_v[idx], color='b', s=20, zorder=2)
        # ax.legend((init, update, pred, gt), ('Init', 'Intermediate', 'Pred', 'GT'),
        #           frameon=False, fontsize=14, labelcolor='r', loc=2)
        # loc=1: upper right
        # loc=3: lower left

        # if args.rotation_range>0:
        init = ax.quiver(A/2, A/2, 1, 1, angles=90, color='r', zorder=2)
        # update = ax.quiver(shift_lons[idx, :], shift_lats[idx, :], 1, 1, angles=thetas[idx, :], color='r')
        pred = ax.quiver(shift_lats[idx, -1], shift_lons[idx, -1], 1, 1, angles=thetas[idx, -1] + 90, color='g', zorder=2)
        gt = ax.quiver(gt_u[idx], gt_v[idx], 1, 1, angles=gt_theta[idx] + 90, color='b', zorder=2)
        # ax.legend((init, pred, gt), ('pred', 'Updates', 'GT'), frameon=False, fontsize=16, labelcolor='r')
        #
        # # for i in range(shift_lats.shape[1]-1):
        # #     ax.quiver(shift_lons[idx, i], shift_lats[idx, i], shift_lons[idx, i+1], shift_lats[idx, i+1], angles='xy',
        # #               color='r')
        #
        ax.axis('off')

        plt.savefig(os.path.join(save_dir, 'points_' + str(loop * B + idx) + '.png'),
                    transparent=True, dpi=A, bbox_inches='tight')
        plt.close()

        grd = transforms.functional.to_pil_image(grd_img[idx], mode='RGB')
        grd.save(os.path.join(save_dir, 'grd_' + str(loop * B + idx) + '.png'))

        sat = transforms.functional.to_pil_image(sat_img[idx], mode='RGB')
        sat.save(os.path.join(save_dir, 'sat_' + str(loop * B + idx) + '.png'))
