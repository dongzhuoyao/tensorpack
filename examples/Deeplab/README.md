
## DeepLab

more detailed results can be seen in [NOTE.md](NOTE.md)

### Result On Pasal VOC
MSF on pascalvoc MSF all tild size: 321*321

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,4GPU,bs32) | 73.65%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8) | 69.3%
### Result On Cityscapes

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF(321*321),1GPU,bs8, 321*321) | 63.6
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF(1024*2048),1GPU,bs1,full) | 66.52