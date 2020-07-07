# CVPR2020 Pose-guided Visible Part Matching for Occluded Person ReID
This is the pytorch implementation of the CVPR2020 paper *"Pose-guided Visible Part Matching for Occluded Person ReID"*

## Dependencies
-Python2.7\
-Pytorch 1.0\
-Numpy

## Related Project
Our code is based on [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid). We adopt [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract pose landmarks and part affinity fields.

## Dataset Preparation
Download the raw datasets [Occluded-REID, P-DukeMTMC-reID](https://github.com/tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets), and [Partial-Reid](https://pan.baidu.com/s/1VhPUVJOLvkhgbJiUoEnJWg) (code:zdl8) which is released by [Partial Person Re-identification](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Zheng_Partial_Person_Re-Identification_ICCV_2015_paper.html). Instructions regarding how to prepare [Market1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) datasets can be found [here](https://kaiyangzhou.github.io/deep-person-reid/datasets.html). And then place them under the directory like:

```
PVPM_experiments/data/
├── ICME2018_Occluded-Person-Reidentification_datasets
│   ├── Occluded_Duke
│   └── Occluded_REID
├── Market-1501-v15.09.15
└── Partial-REID_Dataset
```

## Pose extraction
Install openopse as described [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose).\
Change path to your own dataset root and run sh files in /scripts:
```
sh openpose_occluded_reid.sh
sh openpose_market.sh
``` 
Extracted Pose information can be found [here](https://pan.baidu.com/s/1Majze1iFo7FytREijmQO5A)(code:iwlz)

## To Train PCB baseline

``` 
python scripts/main.py --root PATH_TO_DATAROOT \
 -s market1501 -t market1501\
 --save-dir PATH_TO_EXPERIMENT_FOLDER/market_PCB\
 -a pcb_p6 --gpu-devices 0 --fixbase-epoch 0\
 --open-layers classifier fc\
 --new-layers classifier em\
 --transforms random_flip\
 --optim sgd --lr 0.02\
 --stepsize 25 50\
 --staged-lr --height 384 --width 192\
 --batch-size 32 --base-lr-mult 0.5
```
## To train PVPM
```
python scripts/main.py --load-pose --root PATH_TO_DATAROOT
 -s market1501\
 -t occlusion_reid p_duke partial_reid\
 --save-dir PATH_TO_EXPERIMENT_FOLDER/PVPM\
 -a pose_p6s --gpu-devices 0\
 --fixbase-epoch 30\
 --open-layers pose_subnet\
 --new-layers pose_subnet\
 --transforms random_flip\
 --optim sgd --lr 0.02\
 --stepsize 15 25 --staged-lr\
 --height 384 --width 128\
 --batch-size 32\
 --start-eval 20\
 --eval-freq 10\
 --load-weights PATH_TO_EXPERIMENT_FOLDER/market_PCB/model.pth.tar-60\
 --train-sampler RandomIdentitySampler\
 --reg-matching-score-epoch 0\
 --graph-matching
 --max-epoch 30
 --part-score
```
Trained PCB model and PVPM model can be found [here](https://pan.baidu.com/s/16lr8m-wv-XOXACqIthC8lw)(code:64zy)

# Citation
If you find this code useful to your research, please cite the following paper:
>@inproceedings{gao2020pose,  
  title={Pose-guided Visible Part Matching for Occluded Person ReID},  
  author={Gao, Shang and Wang, Jingya and Lu, Huchuan and Liu, Zimo},  
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},  
  pages={11744--11752},  
  year={2020}  
}





