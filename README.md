# Weakly-supervised Camera Localization by Ground-to-satellite Image Registration

![Framework](./Framework.png)

# Abstract
The ground-to-satellite image matching/retrieval was initially proposed for city-scale ground camera localization. Recently, more and more attention has been paid to increasing the camera pose accuracy by ground-to-satellite image matching, once a coarse location and orientation has been obtained from the city-scale retrieval.  This paper addresses the same scenario. 
However, existing learning-based methods for solving this task require accurate GPS labels of ground images for network training. Unfortunately, obtaining such accurate GPS labels is not always possible, often requiring an expensive RTK setup and suffering from signal occlusion, multi-path signal disruptions, \etc. To address this issue, this paper proposes a weakly-supervised learning strategy for ground-to-satellite image registration. It does not require highly accurate ground truth (GT) pose labels for ground images in the training dataset. Instead, a coarse location and orientation label, either derived from the city-scale retrieval or noisy sensors (GPS, compass, \etc), is sufficient. Specifically, we present a pseudo image pair creation strategy for cross-view rotation estimation network training, and a novel method that leverages deep metric learning for translation estimation between ground-and-satellite image pairs. Experimental results show that our weakly-supervised learning strategy achieves the best performance on cross-area evaluation, compared to the recent state-of-the-art methods that require accurate pose labels for supervision, and shows comparable performance on same-area evaluation.  

### Experiment Dataset
We use two existing dataset to do the experiments: KITTI and VIGOR. For our collected satellite images for KITTI, please first fill this [Google Form](https://forms.gle/Bm8jNLiUxFeQejix7), we will then send you the link for download. 

- **KITTI**: Please first download the raw data (ground images) from http://www.cvlibs.net/datasets/kitti/raw_data.php, and store them according to different date (not category). Your dataset folder structure should be like: 


      raw_data:

        2011_09_26:

          2011_09_26_drive_0001_sync:

            image_00:

        image_01:

        image_02:

        image_03:

        oxts:

          ...

        2011_09_28:

        2011_09_29:

        2011_09_30:

        2011_10_03:

      satmap:

        2011_09_26:

        2011_09_29:

        2011_09_30:

        2011_10_03:

- **VIGOR**: please refer to the following two github pages:
  https://github.com/Jeff-Zilence/VIGOR.git
  https://github.com/tudelft-iv/SliceMatch.git

### Codes

1. Training on KITTI:

    python train_kitti_3DoF.py --rotation_range 10 --stage 0 --level 1
   
    python train_kitti_3DoF.py --rotation_range 10 --stage 1 --share 1
   
    python train_kitti_3DoF.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1
   
    python train_kitti_3DoF.py --rotation_range 10 --stage 1 --share 1 --ConfGrd 1 --level 1 --GPS_error_coe 1

3. Training on VIGOR:
    
    python train_vigor_2DoF.py --rotation_range 0 --share 0 --ConfGrd 1 --level 1 --Supervision Weakly --area same
   
    python train_vigor_2DoF.py --rotation_range 0 --share 0 --ConfGrd 1 --level 1 --Supervision Weakly --area cross
   
    python train_vigor_2DoF.py --rotation_range 0 --share 0 --ConfGrd 1 --level 1 --Supervision Weakly --area same  --GPS_error_coe 1
   
    python train_vigor_2DoF.py --rotation_range 0 --share 0 --ConfGrd 1 --level 1 --Supervision Weakly --area cross --GPS_error_coe 1
   
    python train_vigor_2DoF.py --rotation_range 180 --share 0 --ConfGrd 1 --level 1 --Supervision Weakly --area same
   
    python train_vigor_2DoF.py --rotation_range 180 --share 0 --ConfGrd 1 --level 1 --Supervision Weakly --area cross
   
    python train_vigor_2DoF.py --rotation_range 180 --share 0 --ConfGrd 1 --level 1 --Supervision Weakly --area same --GPS_error_coe 1
   
    python train_vigor_2DoF.py --rotation_range 180 --share 0 --ConfGrd 1 --level 1 --Supervision Weakly --area cross --GPS_error_coe 1
   
   
   
2. Evaluation:

    Plz simply add "--test 1" after the training commands. E.g. 

    python train_kitti_3DoF.py --test 1



### Models:
Our trained models are available [here](https://anu365-my.sharepoint.com/:f:/g/personal/u6293587_anu_edu_au/EqbaNmTp8lZHu5xoZ9gt7ycB7Us_22izmh7BImRQETpjKw?e=ZDQ2Vb). 



### Publications
This work is accepted by ECCV 2024.  


