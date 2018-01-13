
## DeepLab

more experimental results on pascalVOC can be seen in [NOTE.md](NOTE.md)


more experimental results on RemoteSense Images can be seen in [aerial/README.md](aerial/README.md)


### Result On Pasal VOC

MSF on pascalvoc MSF all tild size: 321*321

MSC+Coco+Aug+ASPP+CRF=77.69% mIoU

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8) | **69.3%**
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,4GPU,bs32) | 73.65%

if without imgaug.RandomResize, the mIoU will be 68.4% rather than 69.3%.

### Result On Cityscapes

Full+Aug+ASPP+CRF=71.4% mIoU

Arch | paper mIoU |Val mIoU
------------ | -------- | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF(1024X2048),1GPU,bs1,full) | 66.6(Full Image)  | **66.52**
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF(321X321),1GPU,bs8, 321*321) | -- | 63.6

### TODO

* WideResNet
* ResNext on cifar10(which in details: ResNeXt-29, 8×64d with baseline wideResNet)
* SE-ResNeXt-101+Mixup(currently best basic model) on cifar10
* GCN
* FRRN
* DCN
