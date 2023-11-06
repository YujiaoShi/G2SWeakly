
#CUDA_VISIBLE_DEVICES=0 python train_ford_rot_corr.py --rotation_range 10 --stage 0 --share 0
#CUDA_VISIBLE_DEVICES=0 python train_ford_rot_corr.py --rotation_range 10 --stage 1 --share 1
#CUDA_VISIBLE_DEVICES=0 python train_ford_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1
#CUDA_VISIBLE_DEVICES=2 python train_ford_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --train_whole 1
#CUDA_VISIBLE_DEVICES=1 python train_ford_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --GPS_error_coe 1 --visualize 1 --train_log_end 5


#CUDA_VISIBLE_DEVICES=0 python train_ford_rot_corr.py --rotation_range 10 --stage 0 --share 0 --train_log_start 0 --train_log_end 1
#CUDA_VISIBLE_DEVICES=0 python train_ford_rot_corr.py --rotation_range 10 --stage 0 --share 0 --train_log_start 1 --train_log_end 2
#CUDA_VISIBLE_DEVICES=0 python train_ford_rot_corr.py --rotation_range 10 --stage 0 --share 0 --train_log_start 2 --train_log_end 3
#CUDA_VISIBLE_DEVICES=0 python train_ford_rot_corr.py --rotation_range 10 --stage 0 --share 0 --train_log_start 3 --train_log_end 4
#CUDA_VISIBLE_DEVICES=0 python train_ford_rot_corr.py --rotation_range 10 --stage 0 --share 0 --train_log_start 4 --train_log_end 5
#CUDA_VISIBLE_DEVICES=0 python train_ford_rot_corr.py --rotation_range 10 --stage 0 --share 0 --train_log_start 5 --train_log_end 6


#CUDA_VISIBLE_DEVICES=0 python train_ford_rot_corr.py --rotation_range 10 --stage 1 --share 1 --train_log_start 0 --train_log_end 1

#python train_ford_3DoF_Fully.py --batch_size 4 --train_log_end 5 --test 1 --test_log_ind 0
#python train_ford_3DoF_Fully.py --batch_size 4 --test 1 --test_log_ind 1 --train_log_end 5
#python train_ford_3DoF_Fully.py --batch_size 4 --test 1 --test_log_ind 2 --train_log_end 5
#python train_ford_3DoF_Fully.py --batch_size 4 --test 1 --test_log_ind 3 --train_log_end 5
#python train_ford_3DoF_Fully.py --batch_size 4 --test 1 --test_log_ind 4 --train_log_end 5
#python train_ford_3DoF_Fully.py --batch_size 4 --test 1 --test_log_ind 5 --train_log_end 5

#CUDA_VISIBLE_DEVICES=0 python train_ford_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --train_log_start 0 --train_log_end 1
#CUDA_VISIBLE_DEVICES=0 python train_ford_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --train_log_start 1 --train_log_end 2
#CUDA_VISIBLE_DEVICES=0 python train_ford_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --train_log_start 2 --train_log_end 3
#CUDA_VISIBLE_DEVICES=1 python train_ford_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --train_log_start 3 --train_log_end 4
#CUDA_VISIBLE_DEVICES=1 python train_ford_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --train_log_start 4 --train_log_end 5
#CUDA_VISIBLE_DEVICES=0 python train_ford_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --train_log_start 5 --train_log_end 6

CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.3;
CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.6;
CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.9;
CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.6 --GPS_error_coe 1;
CUDA_VISIBLE_DEVICES=1 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.9 --GPS_error_coe 1;
