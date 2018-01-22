
## DeepLab

more experimental results on pascalVOC can be seen in [NOTE.md](NOTE.md)


more experimental results on RemoteSense Images can be seen in [aerial/README.md](aerial/README.md)


### Result On Pasal VOC

MSF on pascalvoc MSF all tild size: 321*321

MSC+Coco+Aug+ASPP+CRF=77.69% mIoU

Arch | Val mIoU
------------ | -------------
deeplabv2.voc.imagesize473| **71.5%** 


if without imgaug.RandomResize, the mIoU will be 68.4% rather than 69.3%.

### Result On Cityscapes

Full+Aug+ASPP+CRF=71.4% mIoU

Arch  |Val mIoU
------------ | -------- 
deeplabv2.cs.imagesize672.scale18.py| **74.21%**

## BN strategy

* standard gamma,beta are updated in 4 gpus. please note that mean,variance doesn't involve in back propagation.
* EMA of the mean, variance is maintained in the main_training_tower. 
* when back propagation, the gamma, beta are averaged over 4gpus just like other variable.
* when inference, we use the EMA of mean, variance in main_training_tower, and use the averaged gamma,beta.
* when fine tuning to segmentation task, all gamma, beta are trained,(in this case you must make the batch size as large as possible to gain a stable BN statistics.)

### TODO

* DeformableConvolution [https://github.com/Zardinality/TF-deformable-conv](https://github.com/Zardinality/TF-deformable-conv)
* WideResNet
* ResNext on cifar10(which in details: ResNeXt-29, 8×64d with baseline wideResNet)
* SE-ResNeXt-101+Mixup(currently best basic model) on cifar10
* GCN
* FRRN
* DCN
