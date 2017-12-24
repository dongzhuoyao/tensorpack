
# Experiment on [https://project.inria.fr/aerialimagelabeling/](https://project.inria.fr/aerialimagelabeling/)


uploaded model on val result: 87.25%

### models

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,no MSF,1GPU,bs10) | 87.17%,acc:95.58



### How Multi scale Fusion influence the result?

Arch | Val mIoU
------------ | -------------
(0.8,0.9,1,1.1,1.2) | 88.27
(0.9,1,1.1) | 88.1
1           | 87.1

