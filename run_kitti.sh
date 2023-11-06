
#CUDA_VISIBLE_DEVICES=2 python train_kitti_rot_corr.py --rotation_range 10 --stage 0 --share 0 --level 1
#CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1
CUDA_VISIBLE_DEVICES=2 python train_kitti_3DoF.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --test 1
CUDA_VISIBLE_DEVICES=2 python train_kitti_3DoF.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --GPS_error_coe 1 --test 1
#
#
#
#CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 40 --stage 1 --share 1 --ConfGrd 1 --train_noisy 1
#CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 40 --stage 1 --share 1 --ConfGrd 1 --train_noisy 1 --GPS_error_coe 1


#CUDA_VISIBLE_DEVICES=2 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.2;
#CUDA_VISIBLE_DEVICES=2 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.7;
#CUDA_VISIBLE_DEVICES=2 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.8;
#CUDA_VISIBLE_DEVICES=2 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.3 --GPS_error_coe 1;
#CUDA_VISIBLE_DEVICES=2 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.4 --GPS_error_coe 1;
#CUDA_VISIBLE_DEVICES=2 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.5 --GPS_error_coe 1;

#CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.3;
#CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.6;
#CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.9;
#CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.6 --GPS_error_coe 1;
#CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.9 --GPS_error_coe 1;


#CUDA_VISIBLE_DEVICES=0 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.4;
#CUDA_VISIBLE_DEVICES=0 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.5;
#CUDA_VISIBLE_DEVICES=0 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.2 --GPS_error_coe 1;
#CUDA_VISIBLE_DEVICES=0 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.7 --GPS_error_coe 1;
#CUDA_VISIBLE_DEVICES=0 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.8 --GPS_error_coe 1;



#CUDA_VISIBLE_DEVICES=0 python train_kitti_rot_corr.py --rotation_range 20 --stage 0 --share 0
#

#CUDA_VISIBLE_DEVICES=0 python train_kitti_rot_corr.py --shift_range_lat 5 --shift_range_lon 5 --stage 0  --level 1
#CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --shift_range_lat 5 --shift_range_lon 5 --stage 1  --level 1 --share 1 --ConfGrd 1 --test 1 --GPS_error_coe 1
#CUDA_VISIBLE_DEVICES=0 python train_kitti_rot_corr.py --shift_range_lat 10 --shift_range_lon 10 --stage 0  --level 1
#CUDA_VISIBLE_DEVICES=2 python train_kitti_rot_corr.py --shift_range_lat 10 --shift_range_lon 10 --stage 1  --level 1 --share 1 --ConfGrd 1 --test 1 --GPS_error_coe 1
#
#
## ablation for rebuttal:
##CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 0 --ConfGrd 1
#
##sleep 140m
#
##CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --proj_img 1
#
##CUDA_VISIBLE_DEVICES=2 python train_kitti_rot_corr.py --rotation_range 10 --stage 0 --share 0 --proj_img 1
##
##CUDA_VISIBLE_DEVICES=2 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --GPS_error_coe 1 --contrastive_coe 0
#
#CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 40 --stage 0 --level 1
#CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 40 --stage 1 --share 1 --ConfGrd 1 --level 1
#
#CUDA_VISIBLE_DEVICES=2 python train_kitti_rot_corr.py --rotation_range 20 --stage 0 --level 1
#CUDA_VISIBLE_DEVICES=2 python train_kitti_rot_corr.py --rotation_range 20 --stage 1 --share 1 --ConfGrd 1 --level 1
