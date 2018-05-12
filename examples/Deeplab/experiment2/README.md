

## DeepLab


Deeplabv1: [http://ethereon.github.io/netscope/#/gist/f3643b581154e8d6f26bde092e8363ad](http://ethereon.github.io/netscope/#/gist/f3643b581154e8d6f26bde092e8363ad)

Deeplabv2: [http://ethereon.github.io/netscope/#/gist/ff41a61a5384146fc099511a5075e1f9](http://ethereon.github.io/netscope/#/gist/ff41a61a5384146fc099511a5075e1f9)

Totally, PascalVOC 30K iterations, Cityscapes 45K iterations.

Pascal VOC best scale(500 times/image):

Batch Size | Scale
------------ | -------------
15| 50

Arch|batch/image_size| #params| size |speed
--| --| --| ----| ----
DenseNet(L121-k32)|15/321| 6.9M|26.65MB|2.8iter/s
DenseNet(L201-k32)|15/321| 18M|69.24MB|1.52iter/s
DenseNet(L264-k32)|7/321| 30M|117.22MB|2iter/s
resnet101|25/321|42M|163MB|1.4iter/s|



### Pascal VOC

Arch | #Params(M) |Val mIoU 
------------ | -------------| -------------
deeplabv2.voc.imagesize473|42|70.5
deeplabv2.voc.imagesize473.scratch.lr1e-1||very low
deeplabv2.voc.imagesize473.scratch.lr1e-2|| 44.1
deeplabv2.voc_train.imagesize321.scale50.scratch||epoch5:27,terminated
deeplabv2.voc.imagesize473.scratch|42|49.9
deeplabv2.voc.scale16.imagesize473.scratch|42|54.6
deeplabv2.voc.imagesize321.densenet31k48.scratch|5|48.6
deeplabv2.voc.imagesize321.densenet31k60.scratch|8|49.4
deeplabv2.voc.imagesize321.densenet121k48.scratch|15|50.6
deeplabv2.voc.imagesize321.densenet121k60.scratch|24|50.2
deeplabv2.voc.imagesize321.scratch|42|50.1


### Camvid 
Arch | #Params(M) |Test mIoU(66)
-------- | -------------| -----------
deeplabv2.camvid.imagesize473.scratch_acturally321||(val:62.5) 49
deeplabv2.camvid.imagesize473.scratch|47|(val:62.5)
deeplabv2.camvid.imagesize473.res50.scratch|23|(val:56.7)
--|--|--
deeplabv2.camvid.imagesize473.toy.realtest.scratch|2.9|**38**
deeplabv2.camvid.imagesize473.toy.resnet101.scratch|42.7|49.1
deeplabv2.camvid.imagesize473.toy.densenet121k32.realtest.scratch|6.9|44.3
deeplabv2.camvid.imagesize473.toy.nocompression.realtest.scratch|5.2|44.1
deeplabv2.camvid.imagesize473.toy.stem.realtest.scratch|3.2|42.9
deeplabv2.camvid.imagesize473.toy.k60.realtest.scratch|8.2|38
------------ | -------------| -------------
deeplabv2.camvid.imagesize473.toy.newbaseline.realtest.scratch|2.9|38
deeplabv2.camvid.imagesize473.toy.newbaseline.removeLatterPooling.realtest.scratch|2.9|44.1
deeplabv2.camvid.imagesize321.toy.realtest.scratch|2.9|50
deeplabv2.camvid..imagesize321.toy.newbaseline.dense30k36.py|2.9|52.7
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k36|5.7|52
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k36.senet(ratio=8)|5.7|52.3
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k36.senet1|5.8|52.7
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k36.senet4|5.7|53.6
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k36.senet4.did|5.9|53.2
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k36.senet4.did4|6.1|52.2
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k48|9.6|54.8
deeplabv2.camvid..imagesize360x480.toy.newbaseline.stem.nocompression.nopooling.dense30k48|9.6|57.2
deeplabv2.camvid..imagesize360x480.toy.newbaseline.stem.nocompression.nopooling.dense30k48.batchsize16|9.6|56.8
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense42k48|15.9|55
deeplabv2.camvid..imagesize225.toy.newbaseline.stem.nocompression.nopooling.dense30k48||52.6
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k48.nodropout|9.6|54
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k48.senet4|9.69|53.7
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k48.newdid|10.7|55.1
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k48.newdid2|13.6|54.6
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k48.newdid4|14|57.9
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k48.newdid4_1|?|59
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k48.newdid6|11.6|59.2|
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k48.newdid6_senet||59.1
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k48.transitionsenet2|12.8|54.8
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k48.transitionsenet4|10.76|53.5
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k36(actually48).senet4|5.7|53.1
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k36(actually48).senet4.did|12.1|54.4
deeplabv2.camvid.imagesize321.toy.resnet50.scratch|23.7|51.7
deeplabv2.camvid.imagesize321.toy.baselline.stem.nocompression.nopooling.dense121k24.senet4|7.6|53.4
------------ | -------------| -------------
camvid.scratch.imagesize473.stem.nocompression.nopooling.dense30k36||50,termiated
camvid.scratch.imagesize473.densenet30k48||50,terminated
camvid.scratch.imagesize473.densenet30k36||50,terminated
camvid.scratch.imagesize321.densenet30k36|2.9|56
camvid.scratch.imagesize321.stem.nocompression.nopooling.dense121k60|7.6|?
camvid.scratch.imagesize321.stem.nocompression.nopooling.dense30k36|5.7|60
camvid.scratch.imagesize321.stem.compression.nopooling.dense30k36|3.2|58.8
camvid.scratch.imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k48.newdid6.scale640||65.6
camvid.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.3gpu||61.7(why)
camvid.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.scale320.lrx1.4gpu||57~，very low
camvid.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.4gpu.2ndtime.regularization2e-4.scale640||epoch5:21,very low
camvid.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.3gpu.2ndtime.regularization2e-4||65.8
deeplabv2.camvid.imagesize321.res100.noscratch.on_trainval.scale160||61.6
------------ | -------------| -------------
deeplabv2.camvid.imagesize321.res100.scratch.on_trainval.scale32||55.7|
deeplabv2.camvid.imagesize321.res100.scratch.on_trainval.scale160||58.6|



321 model result is  61.2, when validation with 360x480 the result is only 56.1

### Aerial 

Arch |#Params| Val mIoU(target: )
------------ | -------------| -------------
deeplabv2.aerial.imagesize473.scratch||86.72
deeplabv2.aerial.imagesize473.res50.scratch||86.9
camvid.aerial.imagesize473.stem.nocompression.nopooling.dense30k48||88.3|
camvid.aerial.imagesize473.stem.nocompression.nopooling.dense30k48.scale50||88.8|
camvid.aerial.imagesize473.stem.nocompression.nopooling.dense30k48.newdid4||89.1|
camvid.aerial.imagesize473.stem.nocompression.nopooling.dense30k48.newdid4_1||89.0|
camvid.aerial.imagesize473.stem.nocompression.nopooling.dense30k48.newdid6.4gpu_bugfix||90.0
camvid.aerial.imagesize473.stem.nocompression.nopooling.dense30k48.newdid6||89


### Cityscapes(5.2)
Arch |#Params| Val mIoU(target: 69.45, 71.8)
------------ | -------------| -------------
cs.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.fixDim(8)||65.5,(val500:67.93,test:65.9)
cs.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.fixDim.sublinear||64.15
cs.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.fixDim.scale12||70.7,(val500:70.57,test:68.75)
cs.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.fixDim.scale16||72.1(epoch9)
cs.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.fixDim.scale20||72.3(test:70.6)
cs.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.fixDim.scale16.stride8.imgsize800||68.7|
cs.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.fixDim.scale8.stride8.imgsize1024||57.9|
------------ | -------------| -------------
camvid.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.scale1000.fixDim||66.1|
camvid.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.scale800.fixDim.sublinear||62.6(epoch9)
camvid.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.stride8||66|
camvid.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.stride8.scale480|66.3|
deeplabv2.camvid..imagesize321.toy.newbaseline.stem.nocompression.nopooling.dense30k48.newdid6.sublinear.lr1p5||57.88
pascal.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.fixDim.scale4||47|
pascal.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.fixDim.scale8||54|
pascal.scratch.imagesize321.stem.nocompression.nopooling.dense30k48.newdid6.fixDim.scale10||54.9|

### Pascal in Slim
Params | Val mIoU
------------ | -------------
slim.deeplabv2.voc.imagesize473| 58.8
deeplabv3.voc.imagesize473|63.2
deeplabv3.voc.imagesize473|**71.6**
deeplabv3.voc.imagesize473.scale12|70.4
deeplabv3.voc.imagesize513.scale4.lrmultfix.noASPP.fortune|68.8
deeplabv3.voc.imagesize513.scale4.lrmultfix.fortune|71.2
deeplabv3.voc.imagesize513.scale4.lrmultfix.tune|68.9(slowly move down)
deeplabv3.voc.imagesize513.scale4.lrmultfix.tune.betagammafix|69.7(slowly move down then up)
slim.deeplabv2.dilation6.voc.imagesize473|**70.6**



### Result On Cityscapes(offical: deeplabv2 val mIoU:71.4% With Full+Aug+ASPP+CRF)

![cs.png](cs.png)


Arch | Val mIoU
------------ | -------------
deeplabv2.cs.imagesize672.scale18|**74.21** ([test:72.67](https://www.cityscapes-dataset.com/evaluation-results/?submissionID=984))




**notice:** The validation mIOU for this repo is  all achieved  with multi-scale and left-right flippling.

## Devils in version3

* image size=513; batch size=16;

* The proposed model is trained with output stride = 16, and then during inference we apply output stride = 8 to get more detailed feature map. 

* multi-GPU training;

* After training on the trainaug set with 30K iterations and initial learning rate = 0.007, we then freeze batch normalization
parameters, employ output stride = 8(via dilation to control it), and train on the official PASCAL VOC 2012 trainval set for another 30K iterations and smaller base learning rate = 0.001.

* how to evaluate

Arch | Pre | After
------------ | -------------| -------------
deeplabv3.voc.imagesize513.scale4.lrmultfix.noASPP.fortune| 68.9 |59 |


## Ideas

*  Semi Supervised Semantic Segmentation Using Generative Adversarial Network
*  Embedding human skeleton to furthur optimize human segmentation
*  [Semantic Image Synthesis via Adversarial Learning](https://github.com/dongzhuoyao/pytorchgo/tree/master/example/SISviaAL)
*  Reinforcement learning
*  embed traditional method
*  [Semantic-aware  Urban Scene Adaption](https://github.com/Peilun-Li/SG-GAN)
* setup a few-shot segmentation dataset.
* [Weakly Supervised Semantic Segmentation Based on Web Image Co-segmentation](https://ascust.github.io/WSS/)
* Instance-Level Salient Object Segmentation
* [Learning Affinity via Spatial Propagation Networks](https://arxiv.org/abs/1710.01020)
*  [A Classification Refinement Strategy for Semantic Segmentation](https://arxiv.org/abs/1801.07674)
* [Statistically Motivated Second Order Pooling](https://arxiv.org/abs/1801.07492)
* [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action
Recognition](https://arxiv.org/abs/1801.07455)
*  [LIFT](https://github.com/cvlab-epfl/tf-lift)
* use [Pose-Guided-Person-Image-Generation](https://github.com/charliememory/Pose-Guided-Person-Image-Generation) to realize human segmentation

## Target

*  slim deeplabv2 pascal reproduce
*  slim deeplabv2 Cityscapes reproduce
*  slim deeplabv2 Mapillary reproduce
* larger batch size use gradient checkpoint
