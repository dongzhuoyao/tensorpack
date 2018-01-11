
## DeepLab

**check augmentation bugs**


### sobel FPN

Arch | Val mIoU
------------ | -------------
deeplabv2.naked.fpn.standard(lr10,scale8) |67.8%,66.5%(second time)|
deeplabv2.naked.fpnstandard.learnable.sobel.channelwise(lr10)|67.8%|
deeplabv2.naked.fpn.standard.scale10(lr10)|67.1|
deeplabv2.naked.fpn.standard.scale10(lr12)|--|
------------ | -------------
deeplabv2.naked.fpn(lr10,scale8)) 165.41MB | 68.3%(strange result, too high), 65.4%(second time ),66.8%(third time)|
deeplabv2.naked.fpn.learnable.sobel.channelwise(lr10)|67.8%,66.9%(second time)|
deeplabv2.naked.fpn.bilinear(lr10)|66.2%|
deeplabv2.naked.fpn.learnable.sobel.channelwise.1order(lr10)| 68.2%,68.4%(second time)|
deeplabv2.cs.naked.fpn(lr10)) 165.41MB |55.2|
deeplabv2.cs.naked.fpn.learnable.sobel.channelwise.1order(lr10)|57.0|
deeplabv2.naked.fpn.scale10(lr10)) 165.41MB |68.1|
deeplabv2.naked.fpn.scale12(lr10)) 165.41MB |66.2|
------------ | -------------
deeplabv2.naked.fpn.lr1,165.41MB|63.4%|
deeplabv2.naked.fpn.learnable.sobel.channelwise.lr1,165.41MB|67.3%|
deeplabv2.naked.fpn.learnable.sobel.channelcross.lr1|65.1%|


### GCN

Arch |official Val mIoU |Val mIoU
------------ | -------------|----------
deeplabv2.naked.gcn(lr10)| 74.1 with resnet150,coco+pascal+BSDS |66.8(buggy)|



### trimap

Arch | Val mIoU
------------ | -------------
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,trimap) | 60.26%
Deeplabv2-resnet101-cs(edgemix) | non-converge
Deeplabv2-resnet101-pascal(edgemix) | non-converge

### sobel

Arch | Val mIoU
------------ | -------------
deeplabv2.naked | baseline: **70%**
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,edge) | 69.49%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,edgescale) | 69.66%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,edge.conv3) | 69.00%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,sobel.lr10) | 68.1%,68.7%
Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,sobel) | 67.4%,69.6%
~~Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,sobel-res234)~~ | non-converge
~~Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,sobellast)~~ | 66.3%
~~Deeplabv2-resnet101(no multi-branch,no ASPP,with MSF,1GPU,bs8,sobel.head)~~ | 50.2%
deeplabv2.naked.edge.learnable.sobel,258MB | 70%
deeplabv2.naked.edge.learnable.sobel.channelwise,163.75MB | **69.9%**
deeplabv2.naked.edge.learnable.sobel.channelwise.lr10,163.75MB | **70.3%**
deeplabv2.naked.edge.learnable.sobel.channelwise.lr1.newcode|71.1%|
deeplabv2.naked.edge.learnable.sobel.channelwise.lr10.newcode|71.3%|





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


## Ideas

*  Semi Supervised Semantic Segmentation Using Generative Adversarial Network
*  Embedding human skeleton to furthur optimize human segmentation
*  [Semantic Image Synthesis via Adversarial Learning](https://github.com/dongzhuoyao/pytorchgo/tree/master/example/SISviaAL)
*  Reinforcement learning
*  embed traditional method
*  [Semantic-aware  Urban Scene Adaption](https://github.com/Peilun-Li/SG-GAN)
* setup a few-shot segmentation dataset.
* [Weakly Supervised Semantic Segmentation Based on Web Image Co-segmentation](https://ascust.github.io/WSS/)


