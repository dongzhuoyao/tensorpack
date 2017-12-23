


### models

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8) | (pointwise-evalution)80.82, 81.32,81.37,81.99,81.45%



### two mIoU evaluation method comparision
Method | Val mIoU
------------ | -------------
from DCN(classwise-evaluation) | 81.45
from myself(pointwise-evaluation) | 87.24



### How Multi scale Fusion influence the result?

Arch | Val mIoU
------------ | -------------
(0.9,1,1.1) | 81.45
1           | 81.9

