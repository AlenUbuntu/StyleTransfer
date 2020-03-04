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

To run LST, PyTorch 0.4.1 version is required. We recommend users to install it in an anaconda virtual environment, since lots of functions in PyTorch 0.4.1 are deprecated. Details about setting and activating the virtual environment is [here]().

## Style Transfer
Modify model settings in the coressponding yaml file (configs/xxx_test.yaml or configs/xxx_train.yaml). Note that lst_spn_train.yaml, lst_spn_test.yaml and fps_photo_test.yaml are for photo-realistic style transfer only.
### Artisitc Style Transfer

* For a single pair test
```sh
python StyleTransfer/tools/test.py --config-file StyleTransfer/configs/xxx_test.yaml --content path/to/content image --style path/to/style image
```
* For large number of pair tests
```sh
python StyleTransfer/tools/test.py --config-file StyleTransfer/configs/xxx_test.yaml --contentDir path/to/content --styleDir path/to/style --mode 1
```
In the second case, we assume the names of paired content and style images are same.

Some examples are given as below:

![](https://github.com/AlenUbuntu/StyleTransfer/blob/master/images/demo2.png)

### Artistic Style Interpolation
StyleTransfer Library also supports the transferring or synthesis of multiple styles thourgh interpolation. 

styleInterpWeights is the flat to specify interpolation weights, i.e., weight of each style image.

```sh
python StyleTransfer/tools/test.py --config-file StyleTransfer/configs/xxx_test.yaml --content /path/to/content image --style /path/to/style1,/path/to/style2,... --styleInterpWeights 10,10,...
```

Below is an example of handling four styles.
```sh
python StyleTransfer/tools/test.py --config-file StyleTransfer/configs/adain_test.yaml --content demo/content/1.jpg --style demo/style/11.jpg,demo/style/12.jpg,demo/style/1.jpg,demo/style/in3.jpg --styleInterpWeights 0,0,0,100
```
![](https://github.com/AlenUbuntu/StyleTransfer/blob/master/images/interpolation.png)


### Spatial Control (Artistic)

## Photo-Realistic Style Transfer 
### Normal Style Transfer
### Spatial Control
