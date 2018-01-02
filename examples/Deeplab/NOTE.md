
## DeepLab

### trimap

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,trimap) | 60.26%
Deeplabv2-resnet101-cs(edgemix) | non-converge
Deeplabv2-resnet101-pascal(edgemix) | non-converge

### sobel

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8) | 69.3%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,edge) | 69.49%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,edgescale) | 69.66%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,edge.conv3) | 69.00%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,sobel.lr10) | 68.1%,68.7%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,sobel) | 67.4%,69.6%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,sobel-res234) | non-converge
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,sobellast) | 66.3%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,sobel.head) | 50.2%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,learnable_sobel),258MB | 70%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,channelwise) | 69.9%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,channelwise.lr10) | 70.3%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,deeplabv2.naked.fpn) 165.41MB | 68.3%
deeplabv2.naked.fpn.learnable.sobel.channelwise|67.8%|
deeplabv2.naked.fpn.bilinear|66.2%|


### Result On Pasal VOC
MSF on pascalvoc MSF all tild size: 321*321

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,4GPU,bs32) | 73.65%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8) | 69.3%
Deeplabv2-resnet101(no multi-branch,no ASPP,no MSF,1GPU,bs8) | 68.6%



### Result On Cityscapes

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF(321x321),1GPU,bs8, 321x321) | 63.6
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF(1024x2048),1GPU,bs1,full) | 66.52


**notice:** The validation mIOU for this repo is  all achieved with multi-scale and left-right flippling.
