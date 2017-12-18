
## DeepLab


### Result On Pasal VOC
MSF on pascalvoc MSF all tild size: 321*321

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,4GPU,bs32) | 73.65%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8) | 69.3%
Deeplabv2-resnet101(no multi-branch,no ASPP,no MSF,1GPU,bs8) | 68.6%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,trimap) | 60.26%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,edge) | 69.49%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,edgescale) | 69.66%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,edge.conv3) | 69.00%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,sobel.lr10) | 67.70%??

### Result On Cityscapes

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF(321x321),1GPU,bs8, 321x321) | 63.6
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF(1024x2048),1GPU,bs1,full) | 66.52

