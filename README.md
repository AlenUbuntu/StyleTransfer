# StyleTransfer:
This is an PyTorch image deep style transfer library. It provies implementations of current SOTA algorithms, including

* **AdaIN** (Artistic)

  [**Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization** (ICCV 2017)](https://arxiv.org/abs/1703.06868)

* **WCT** (Artistic)

  [**Universal Style Transfer via Feature Transforms** (NIPS 2017)](https://arxiv.org/abs/1705.08086)
  
* **LinearStyleTransfer (LST)** (Artistic, Photo-Realistic)

  [**Learning Linear Transformations for Fast Image and Video Style Transfer** (CVPR 2019)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Learning_Linear_Transformations_for_Fast_Image_and_Video_Style_Transfer_CVPR_2019_paper.pdf)

* **FastPhotoStyle (FPS, NVIDIA)** (Photo-Realistic)

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

In addition, we need to compile the [pytorch_spn](https://github.com/Liusifei/pytorch_spn) module.
```sh
cd lib/SPN/pytorch_spn/
sh make.sh
cd ../../../
```

## Style Transfer
Modify model settings in the coressponding yaml file (configs/xxx_test.yaml or configs/xxx_train.yaml). Note that lst_spn_train.yaml, lst_spn_test.yaml and fps_photo_test.yaml are for photo-realistic style transfer only.

```--resize``` flag below is optional that can accelerate computing and save memory.

### Artisitc Style Transfer

* For a single pair test
```sh
python StyleTransfer/tools/test.py --config-file StyleTransfer/configs/xxx_test.yaml --content path/to/content_image --style path/to/style_image [--resize]
```
* For large number of pair tests
```sh
python StyleTransfer/tools/test.py --config-file StyleTransfer/configs/xxx_test.yaml --contentDir path/to/content --styleDir path/to/style --mode 1 [--resize]
```
In the second case, we assume the names of paired content and style images are same.

Some examples are given as below:

![](https://github.com/AlenUbuntu/StyleTransfer/blob/master/images/demo2.png)

### Artistic Style Interpolation
StyleTransfer Library also supports the transferring or synthesis of multiple styles thourgh interpolation. 

styleInterpWeights is the flag to specify interpolation weights, i.e., weight of each style image.

**Note that currently only AdaIN supports style interpolation**

```sh
python StyleTransfer/tools/test.py --config-file StyleTransfer/configs/xxx_test.yaml --content /path/to/content_image --style /path/to/style1_image,/path/to/style2_image,... --styleInterpWeights 10,10,... [--resize]
```

Below is an example of handling four styles.
```sh
python StyleTransfer/tools/test.py --config-file StyleTransfer/configs/adain_test.yaml --content demo/content/1.jpg --style demo/style/11.jpg,demo/style/12.jpg,demo/style/1.jpg,demo/style/in3.jpg --styleInterpWeights 0,0,0,100
```
![](https://github.com/AlenUbuntu/StyleTransfer/blob/master/images/interpolation.png)

### Artistic Spatial Control
The one-click global transfer still does not meet requirements from professinal users (e.g., artists) in many cases. Users prefer to transfer different styles to different regions in the content image, i.e., spatial control. StyleTransfer Library supports this operation.

**Note that currently only AdaIN and WCT supports spatial control**

```sh
python StyleTransfer/tools/test.py --config-file configs/xxx_test.yaml --content /path/to/content_image --style /path/to/style1_image,/path/to/style2_image --mask /path/to/mask_image [--resize]
```

Here, we provide an example of transferring two styles to the foreground and background respectively, i.e., Style I for foreground (mask=1), Style II for background (mask=0), provided a binary mask.

```sh
python tools/test.py --config-file configs/adain_test.yaml --content demo/mask/spatial_content.jpg --style demo/mask/mask_1.jpg,demo/mask/mask_2.jpg --mask demo/mask/mask.png
python tools/test.py --config-file configs/wct_test.yaml --content demo/mask/spatial_content.jpg --style demo/mask/mask_1.jpg,demo/mask/mask_2.jpg --mask demo/mask/mask.png
```
![](https://github.com/AlenUbuntu/StyleTransfer/blob/master/images/demo3.png)

### Photo-Realistic Style Transfer 
FPS generally provides two inference modes: FPS-Fast and FPS-Slow. FPS-Slow utilizes the prpogator (photo smoothing) described in the paper which is computationally expensive and slow. FPS-Fast replace that propogator with Guided Filter proposed by Kaiming He. 

**We found FPS-Fast shows a similar performance as FPS-Slow but is much faster**.

* For a single pair test
```sh
python StyleTransfer/tools/test_photorealistic.py --config-file StyleTransfer/configs/lst_spn_test.yaml --content path/to/content_image --style path/to/style_image [--resize]
```
or 
```sh
python StyleTransfer/tools/test_photorealistic.py --config-file StyleTransfer/configs/fps_photo_test.yaml --content path/to/content_image --style path/to/style_image  [--resize]
```

* For large number of pair tests
```sh
python StyleTransfer/tools/test_photorealistic.py --config-file StyleTransfer/configs/lst_spn_test.yaml --contentDir path/to/content --styleDir path/to/style --mode 1 [--resize]
```
or 
```sh
python StyleTransfer/tools/test_photorealistic.py --config-file StyleTransfer/configs/fps_photo_test.yaml --contentDir path/to/content --styleDir path/to/style --mode 1  [--resize]
```

Some examples are given below:

![](https://github.com/AlenUbuntu/StyleTransfer/blob/master/images/demo4.png)

### Photo-Realistic Spatial Control

## TODO

* LST: support style interpolation and spatial control
* WCT: support style interpolation
