
## DeepLab


### Result On Pasal VOC

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,4GPU,bs32) | 73.65%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8) | 69.3%
Deeplabv2-resnet101(no multi-branch,no ASPP,no MSF,1GPU,bs8) | 68.6%
Deeplabv2-resnet101(no multi-branch,no ASPP,MSF,1GPU,bs8,trimap) | 60.26%
Deeplabv2-resnet101(no multi-branch,no ASPP,MSF,1GPU,bs8,edge) | 69.49%
Deeplabv2-resnet101(no multi-branch,no ASPP,MSF,1GPU,bs8,edgescale) | 69.66%

### Result On Cityscapes

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,4GPU,bs32) | --

