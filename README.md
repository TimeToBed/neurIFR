# Instance Feature Optimization for Enhancing Few-Shot Classification Performance

Model inference stage code

## Dataset

The official link of CUB-200-2011 is [here](http://www.vision.caltech.edu/datasets/cub_200_2011/). The category splitting of CUB-200-2011 is performed in the same manner as in [Bi-FRN](https://github.com/PRIS-CV/Bi-FRN). When conducting the mini->cub task, the category splitting of CUB-200-2011 is according to the [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot).

The official link of ImageNet is [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php). The tiered-ImageNet dataset is extracted from ImageNet and is divided into training, validation, and test sets according to the [FRN](https://github.com/Tsingularity/FRN) 

- CUB_200_2011 \[[Download Link](https://drive.google.com/file/d/1WxDB3g3U_SrF2sv-DmFYl8LS0p_wAowh/view)\]
- mini-ImageNet \[[Download Link](https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view)\] 

## Train

* The training phase code will be provided after the paper is accepted


## Test

```shell
    python test.py
```

## Trained model weights

We provide trained model weights for IFO with a ResNet-12 network backbone. You can download these:
- Download from Baidu Netdisk [Link](https://pan.baidu.com/s/11rbgsF8IhXfw8yjEjfwHXA?pwd=3icm) and extraction code 3icm


Our work follow MIT License
