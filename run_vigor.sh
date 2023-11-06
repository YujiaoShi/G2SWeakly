
CUDA_VISIBLE_DEVICES=2 python train_vigor_2DoF.py --Supervision Weakly --level 1 --area cross --rotation_range 0 --test 1;
CUDA_VISIBLE_DEVICES=2 python train_vigor_2DoF.py --Supervision Weakly --level 1 --area same --rotation_range 0 --test 1 ;
CUDA_VISIBLE_DEVICES=2 python train_vigor_2DoF.py --Supervision Weakly --level 1 --area cross --rotation_range 180 --test 1 ;
CUDA_VISIBLE_DEVICES=2 python train_vigor_2DoF.py --Supervision Weakly --level 1 --area same  --rotation_range 180 --test 1;


CUDA_VISIBLE_DEVICES=2 python train_vigor_2DoF.py --Supervision Fully --level 1 --area cross --rotation_range 0 --test 1;
CUDA_VISIBLE_DEVICES=2 python train_vigor_2DoF.py --Supervision Fully --level 1 --area same --rotation_range 0 --test 1;
CUDA_VISIBLE_DEVICES=2 python train_vigor_2DoF.py --Supervision Fully --level 1 --area cross --rotation_range 180 --test 1;
CUDA_VISIBLE_DEVICES=2 python train_vigor_2DoF.py --Supervision Fully --level 1 --area same  --rotation_range 180 --test 1;

CUDA_VISIBLE_DEVICES=2 python train_vigor_2DoF.py --Supervision Weakly --GPS_error_coe 1 --level 1 --area cross --rotation_range 0 --test 1;
CUDA_VISIBLE_DEVICES=2 python train_vigor_2DoF.py --Supervision Weakly --GPS_error_coe 1 --level 1 --area same --rotation_range 0 --test 1 ;
CUDA_VISIBLE_DEVICES=2 python train_vigor_2DoF.py --Supervision Weakly --GPS_error_coe 1 --level 1 --area cross --rotation_range 180 --test 1 ;
CUDA_VISIBLE_DEVICES=2 python train_vigor_2DoF.py --Supervision Weakly --GPS_error_coe 1 --level 1 --area same  --rotation_range 180 --test 1;


#CUDA_VISIBLE_DEVICES=0 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.4;
#CUDA_VISIBLE_DEVICES=0 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.5;
#CUDA_VISIBLE_DEVICES=0 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.2 --GPS_error_coe 1;
#CUDA_VISIBLE_DEVICES=0 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.7 --GPS_error_coe 1;
#CUDA_VISIBLE_DEVICES=0 python train_kitti_rot_corr.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --supervise_amount 0.8 --GPS_error_coe 1;
