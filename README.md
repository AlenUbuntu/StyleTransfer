# StyleTransfer:
This is an PyTorch image deep style transfer library. It provies implementations of current SOTA algorithms, including

* **AdaIN**

  [**Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization** (ICCV 2017)](https://arxiv.org/abs/1703.06868)

* **WCT**

  [**Universal Style Transfer via Feature Transforms** (NIPS 2017)](https://arxiv.org/abs/1705.08086)
  
* **LinearStyleTransfer (LST)**

  [**Learning Linear Transformations for Fast Image and Video Style Transfer** (CVPR 2019)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Learning_Linear_Transformations_for_Fast_Image_and_Video_Style_Transfer_CVPR_2019_paper.pdf)

* **FastPhotoStyle (FPS, NVIDIA)**

  [**A Closed-form Solution to Photorealistic Image Stylization** (ECCV 2018)](https://arxiv.org/abs/1802.06474)

The original implementations can be found at [AdaIN](https://github.com/xunhuang1995/AdaIN-style), 
[WCT](https://github.com/Yijunmaverick/UniversalStyleTransfer), 
[LST](https://github.com/sunshineatnoon/LinearStyleTransfer) and [FSP](https://github.com/NVIDIA/FastPhotoStyle).

With this library, as long as you can find your desired style images on web, you can edit your content image with different transferring effects.
![](https://github.com/AlenUbuntu/StyleTransfer/blob/master/images/demo1.png)

## Prerequisites 
* Linux 
* PyTorch 1.4.0/0.4.1
* Nvidia-GPU and CUDA (for training only)

To run LST, PyTorch 0.4.1 version is required. We recommend users to install it in an anaconda virtual environment, since lots of functions in PyTorch 0.4.1 is depricated. Details about setting and activating the virtual environment is [here]().

## Artistic Style Transfer
### Normal Style Transfer

### Style Interpolation (Artistic Only)

### Spatial Control (Artistic)

## Photo-Realistic Style Transfer 
### Normal Style Transfer
### Spatial Control
