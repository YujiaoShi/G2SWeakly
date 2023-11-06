import random

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

import torch
import pandas as pd
import utils
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

# root_dir = '../../dataset/Kitti1' # '../../data/Kitti' # '../Data' #'..\\Data' #
# root_dir = '/home/yujiao/dataset/Kitti1'
root_dir = '/backup/dataset/Kitti1'

test_csv_file_name = 'test.csv'
ignore_csv_file_name = 'ignore.csv'
satmap_dir = 'satmap'
grdimage_dir = 'raw_data'
left_color_camera_dir = 'image_02/data'  # 'image_02\\data' #
right_color_camera_dir = 'image_03/data'  # 'image_03\\data' #
oxts_dir = 'oxts/data'  # 'oxts\\data' #
# depth_dir = 'depth/data_depth_annotated/train/'

GrdImg_H = 256  # 256 # original: 375 #224, 256
GrdImg_W = 1024  # 1024 # original:1242 #1248, 1024
# GrdOriImg_H = 375
# GrdOriImg_W = 1242
num_thread_workers = 8

# train_file = './dataLoader/train_files.txt'
train_file = './dataLoader/train_files.txt'
test1_file = './dataLoader/test1_files.txt'
test2_file = './dataLoader/test2_files.txt'

train_file_noisy = './dataLoader/train_files_noisy.txt'


# def depth_read(filename):
#     # loads depth map D from png file
#     # and returns it as a numpy array,
#     # for details see readme.txt

#     depth_png = np.array(Image.open(filename), dtype=int)
#     # make sure we have a proper 16bit depth map here.. not 8bit!
#     assert(np.max(depth_png) > 255)

#     depth = depth_png.astype(np.float) / 256.
#     depth[depth_png == 0] = -1.
#     return depth


class SatGrdDataset(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10, data_amount=1.):
        self.root = root

        self.meter_per_pixel = utils.get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        # self.shift_range_meters = shift_range  # in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        self.file_name = [file[:-1] for file in file_name[: int(data_amount*len(file_name))]]

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        file_name = self.file_name[idx]
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]



        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        grd_left_depths = torch.tensor([])
        image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            heading = float(content[5])
            heading = torch.from_numpy(np.asarray(heading))

            # ------ Added for weakly-supervised training ------
            GPS = torch.from_numpy(np.asarray([float(content[0]), float(content[1])]))
            # --------------------------------------------------

            left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                         image_no.lower())
            with Image.open(left_img_name, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                GrdOriImg_W, GrdOriImg_H = grd_img_left.size
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                    # if not self.stereo:
                    break

        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                          (1, 0, utils.CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, utils.CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=Image.BILINEAR)
        # the homography is defined on: from target pixel to source pixel
        # now east direction is the real vehicle heading direction

        # randomly generate shift
        gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
        gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction

        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=Image.BILINEAR)

        # randomly generate roation
        theta = np.random.uniform(-1, 1)
        sat_rand_shift_rand_rot = \
            sat_rand_shift.rotate(theta * self.rotation_range)
        # sat_rand_shift_rand_rot.save('sat_rand_shift_rand_rot.png')

        # # ------ Added for weakly-supervised training ------
        # cos = np.cos(-heading)
        # sin = np.sin(-heading)
        # tx = cos * (utils.CameraGPS_shift_left[0] / self.meter_per_pixel + gt_shift_x * self.shift_range_pixels_lon) \
        #      - sin * (utils.CameraGPS_shift_left[1] / self.meter_per_pixel - gt_shift_y * self.shift_range_pixels_lat)
        # ty = sin * (utils.CameraGPS_shift_left[0] / self.meter_per_pixel + gt_shift_x * self.shift_range_pixels_lon) \
        #      + cos * (utils.CameraGPS_shift_left[1] / self.meter_per_pixel - gt_shift_y * self.shift_range_pixels_lat)
        # GPS2NewCenter = torch.from_numpy(np.asarray([tx, ty]))
        # # the new satellite image center's coordinate under (with respect to) the original satellite image (center)
        # # left--> x positive  down --> y positive   center (0, 0)
        # # --------------------------------------------------

        sat_rand_shift_rand_rot_central_crop = TF.center_crop(sat_rand_shift_rand_rot, utils.SatMap_process_sidelength)
        sat_align_cam_central_crop = TF.center_crop(sat_align_cam, utils.SatMap_process_sidelength)

        if self.satmap_transform is not None:
            sat_rand_shift_rand_rot_central_crop = self.satmap_transform(sat_rand_shift_rand_rot_central_crop)
            sat_align_cam_central_crop = self.satmap_transform(sat_align_cam_central_crop)
        
        # gt_corr_x, gt_corr_y = self.generate_correlation_GTXY(gt_shift_x, gt_shift_y, theta)

        return sat_align_cam_central_crop, \
            sat_rand_shift_rand_rot_central_crop, \
            left_camera_k, grd_left_imgs[0], \
            torch.tensor(-gt_shift_x, dtype=torch.float32).reshape(1), \
            torch.tensor(-gt_shift_y, dtype=torch.float32).reshape(1), \
            torch.tensor(theta, dtype=torch.float32).reshape(1), \
            file_name, GPS


class SatGrdDatasetTest(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = utils.get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        # self.shift_range_meters = shift_range  # in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        # np.random.seed(2022)
        # num = len(file_name)//3
        # random.shuffle(file_name)
        # self.file_name = [file[:-1] for file in file_name[:num]]
        self.file_name = [file[:-1] for file in file_name]
        # self.file_name = []
        # count = 0
        # for line in file_name:
        #     file = line.split(' ')[0]
        #     left_depth_name = os.path.join(self.root, depth_dir, file.split('/')[1],
        #                                    'proj_depth/groundtruth/image_02', os.path.basename(file.strip()))
        #     if os.path.exists(left_depth_name):
        #         self.file_name.append(line.strip())
        #     else:
        #         count += 1
        #
        # print('number of files whose depth unavailable: ', count)


    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        line = self.file_name[idx]
        file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]


        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        grd_left_depths = torch.tensor([])
        # image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            heading = float(content[5])
            heading = torch.from_numpy(np.asarray(heading))

            left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                         image_no.lower())
            with Image.open(left_img_name, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                GrdOriImg_W, GrdOriImg_H = grd_img_left.size
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            # left_depth_name = os.path.join(self.root, depth_dir, file_name.split('/')[1],
            #                                'proj_depth/groundtruth/image_02', image_no)

            # left_depth = torch.tensor(depth_read(left_depth_name), dtype=torch.float32)
            # left_depth = F.interpolate(left_depth[None, None, :, :], (GrdImg_H, GrdImg_W))
            # left_depth = left_depth[0, 0]

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
            # grd_left_depths = torch.cat([grd_left_depths, left_depth.unsqueeze(0)], dim=0)

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                    # if not self.stereo:
                    break

        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                          (1, 0, utils.CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, utils.CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=Image.BILINEAR)
        # the homography is defined on: from target pixel to source pixel
        # now east direction is the real vehicle heading direction

        # randomly generate shift
        # gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
        # gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction
        gt_shift_x = -float(gt_shift_x)  # --> right as positive, parallel to the heading direction
        gt_shift_y = -float(gt_shift_y)  # --> up as positive, vertical to the heading direction

        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=Image.BILINEAR)

        # randomly generate roation
        # theta = np.random.uniform(-1, 1)
        theta = float(theta)
        sat_rand_shift_rand_rot = \
            sat_rand_shift.rotate(theta * self.rotation_range)

        sat_rand_shift_rand_rot_central_crop = TF.center_crop(sat_rand_shift_rand_rot, utils.SatMap_process_sidelength)
        sat_align_cam_central_crop = TF.center_crop(sat_align_cam, utils.SatMap_process_sidelength)

        if self.satmap_transform is not None:
            sat_rand_shift_rand_rot_central_crop = self.satmap_transform(sat_rand_shift_rand_rot_central_crop)
            sat_align_cam_central_crop = self.satmap_transform(sat_align_cam_central_crop)

        # gt_corr_x, gt_corr_y = self.generate_correlation_GTXY(gt_shift_x, gt_shift_y, theta)

        return sat_align_cam_central_crop, \
            sat_rand_shift_rand_rot_central_crop, left_camera_k, grd_left_imgs[0], \
            torch.tensor(-gt_shift_x, dtype=torch.float32).reshape(1), \
            torch.tensor(-gt_shift_y, dtype=torch.float32).reshape(1), \
            torch.tensor(theta, dtype=torch.float32).reshape(1), \
            file_name



class DistanceBatchSampler:
    def __init__(self, sampler, batch_size, drop_last, required_dis, file_name):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.required_dis = required_dis
        self.backup = []
        self.backup_location = torch.tensor([])
        self.file_name = file_name

    def check_add(self, cur_location, location_list):
        if location_list.size()[0] > 0:
            dis = utils.gps2distance(cur_location[0], cur_location[1], location_list[:, 0], location_list[:, 1])
            if torch.min(dis) < self.required_dis:
                return False
        return True

    def __iter__(self):
        batch = []
        location_list = torch.tensor([])

        for idx in self.sampler:
            # check the idx gps location, not less than required distance

            # get location
            file_name = self.file_name[idx].strip().split(' ')[0]
            drive_dir = file_name[:38]
            image_no = file_name[38:]
            # oxt: such as 0000000000.txt
            oxts_file_name = os.path.join(root_dir, grdimage_dir, drive_dir, oxts_dir,
                                          image_no.lower().replace('.png', '.txt'))
            with open(oxts_file_name, 'r') as f:
                content = f.readline().split(' ')

                # get location
                cur_location = [float(content[0]), float(content[1])]
                cur_location = torch.from_numpy(np.asarray(cur_location))

                if self.check_add(cur_location, location_list):
                    # add to batch
                    batch.append(idx)
                    location_list = torch.cat([location_list, cur_location.unsqueeze(0)], dim=0)
                else:
                    # add to back up
                    self.backup.append(idx)
                    self.backup_location = torch.cat([self.backup_location, cur_location.unsqueeze(0)], dim=0)

            if len(batch) == self.batch_size:
                yield batch
                batch = []
                location_list = torch.tensor([])

                # pop back up
                remove = []
                for i in range(len(self.backup)):
                    idx = self.backup[i]
                    cur_location = self.backup_location[i]

                    if self.check_add(cur_location, location_list):
                        # add to batch
                        batch.append(idx)
                        location_list = torch.cat([location_list, cur_location.unsqueeze(0)], dim=0)

                        # need remove from backup
                        remove.append(i)

                for i in sorted(remove, reverse=True):
                    if i == len(self.backup) - 1:
                        # last item
                        self.backup_location = self.backup_location[:i]
                    else:
                        self.backup_location = torch.cat((self.backup_location[:i], self.backup_location[i + 1:]))
                    self.backup.remove(self.backup[i])
                # print('left in backup:',len(self.backup),self.backup_location.size())

        if len(batch) > 0 and not self.drop_last:
            yield batch
            print('batched all, left in backup:', len(self.backup), self.backup_location.size())

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore




def load_train_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10,
                    weak_supervise=False, train_noisy=False, stage=1, data_amount=1):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    if train_noisy:

        train_set = SatGrdDataset(root=root_dir, file=train_file_noisy,
                                  transform=(satmap_transform, grdimage_transform),
                                  shift_range_lat=shift_range_lat,
                                  shift_range_lon=shift_range_lon,
                                  rotation_range=rotation_range, data_amount=1)

    else:

        train_set = SatGrdDataset(root=root_dir, file=train_file,
                                  transform=(satmap_transform, grdimage_transform),
                                  shift_range_lat=shift_range_lat,
                                  shift_range_lon=shift_range_lon,
                                  rotation_range=rotation_range, data_amount=data_amount)

    if weak_supervise and stage != 0:
        file_name = train_set.get_file_list()
        bs = DistanceBatchSampler(torch.utils.data.RandomSampler(train_set), batch_size, True,
                                  2 * (max(shift_range_lat, shift_range_lon) * 1.5), file_name)
        train_loader = DataLoader(train_set, batch_sampler=bs, num_workers=num_thread_workers)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_thread_workers, drop_last=False)
    return train_loader


def load_test1_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    # # Plz keep the following two lines!!! These are for fair test comparison.
    # np.random.seed(2022)
    # torch.manual_seed(2022)

    test1_set = SatGrdDatasetTest(root=root_dir, file=test1_file,
                            transform=(satmap_transform, grdimage_transform),
                            shift_range_lat=shift_range_lat,
                            shift_range_lon=shift_range_lon,
                            rotation_range=rotation_range)

    test1_loader = DataLoader(test1_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_thread_workers, drop_last=False)
    return test1_loader


def load_test2_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    # # Plz keep the following two lines!!! These are for fair test comparison.
    # np.random.seed(2022)
    # torch.manual_seed(2022)

    test2_set = SatGrdDatasetTest(root=root_dir, file=test2_file,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range)

    test2_loader = DataLoader(test2_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)
    return test2_loader







