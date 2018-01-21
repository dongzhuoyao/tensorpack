
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


### TODO

* WideResNet
* ResNext on cifar10(which in details: ResNeXt-29, 8×64d with baseline wideResNet)
* SE-ResNeXt-101+Mixup(currently best basic model) on cifar10
* GCN
* FRRN
* DCN
