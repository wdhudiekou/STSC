# STSC
 

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/wdhudiekou/UMF-CMGR/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.6.0-%237732a8)](https://pytorch.org/)


### Semantic-aware Texture-Structure Feature Collaboration for Underwater Image Enhancement [[IEEE ICRA 2022](https://ieeexplore.ieee.org/abstract/document/9812457)]

By Di Wang, Long Ma, Risheng Liu, and Xin Fan

<div align=center>
<img src="https://github.com/wdhudiekou/STSC/blob/master/Fig/network.png" width="80%">
</div>

## Updates
[2022-05-23] Our paper is available online! [[Paper](https://ieeexplore.ieee.org/abstract/document/9812457)]  


## Requirements
- CUDA 10.1
- Python 3.6 (or later)
- Pytorch 1.6.0
- Torchvision 0.7.0
- OpenCV 3.4

## Get start
Please download the pretrained VGG model [MyVGG.pt](https://drive.google.com/file/d/1v67HJre81RrNJbnLmdpspwSsiMkLBSnP/view?usp=sharing) and [vggfeature.pth](https://drive.google.com/file/d/1TUmfNIPT6PIf0sVNl88CZiqtkNOh13jq/view?usp=share_link) and put them into the folder 'pretrain'
BaiduNetdiskDownload pretrained VGG model [MyVGG.pt] (https://pan.baidu.com/s/1pRx5zGLfV2Co0x_BcJOtJQ?pwd=c915) [code: c915] a [vggfeature.pth](https://pan.baidu.com/s/1bfbThbMeErJJYLv693FuSg?pwd=84zk) [code: 84zk] 

1. Check the dataset path in train.py, and then run:
    ```python
       python train.py
   
2. Check the model and image pathes in test_UIEB.py and then run:
    ```python
       python test_UIEB.py

  Please download the [UIEB pretrained model](https://drive.google.com/file/d/1Rt8-8DfX9UdiPj9AmXiTiQEe4iRG-HUE/view?usp=share_link) and put it into folder './checkpoints/'
   
## Dataset
Please download the following datasets:
*   [UIEB](https://ieeexplore.ieee.org/document/8917818)
*   [EUVP](http://irvlab.cs.umn.edu/resources/euvp-dataset)
*   [RUIE](https://ieeexplore.ieee.org/document/8949763)
*   [USOD](https://github.com/xahidbuffon/SVAM-Net)

## Experimental Results

<div align=center>
<img src="https://github.com/wdhudiekou/STSC/blob/master/Fig/UIEB.png" width="95%">
</div>

<div align=center>
<img src="https://github.com/wdhudiekou/STSC/blob/master/Fig/EUVP.png" width="95%">
</div>

<div align=center>
<img src="https://github.com/wdhudiekou/STSC/blob/master/Fig/RUIE.png" width="95%">
</div>

<div align=center>
<img src="https://github.com/wdhudiekou/STSC/blob/master/Fig/USOD.png" width="95%">
</div>

## Citation
```
@InProceedings{Wang_22_ICRA,
  author    = {Di Wang and Long Ma and Risheng Liu and Xin Fan},
  title     = {Semantic-aware Texture-Structure Feature Collaboration for Underwater
               Image Enhancement},
  booktitle = {{IEEE} International Conference on Robotics and Automation {ICRA}},
  pages     = {4592--4598},
  year      = {2022}
}
```