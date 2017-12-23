


### trimap

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8) | 81.32,81.37,81.99,81.45%
Deeplabv2-resnet101-cs(edgemix) | non-converge
Deeplabv2-resnet101-pascal(edgemix) | non-converge

### How Multi scale Fusion influence the result

Arch | Val mIoU
------------ | -------------
(0.9,1,1.1) | 81.45
1           |   --

