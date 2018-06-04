import numpy as np
import os.path as osp
from util import Map

coco_cat2trainid = {
'toilet':62,
'teddy bear':78,
'cup':42,
'bicycle':2,
'kite':34,
'carrot':52,
'stop sign':12,
'tennis racket':39,
'donut':55,
'snowboard':32,
'sandwich':49,
'motorcycle':4,
'oven':70,
'keyboard':67,
'scissors':77,
'airplane':5,
'couch':58,
'mouse':65,
'fire hydrant':11,
'boat':9,
'apple':48,
'sheep':19,
'horse':18,
'banana':47,
'baseball glove':36,
'tv':63,
'traffic light':10,
'chair':57,
'bowl':46,
'microwave':69,
'bench':14,
'book':74,
'elephant':21,
'orange':50,
'tie':28,
'clock':75,
'bird':15,
'knife':44,
'pizza':54,
'fork':43,
'hair drier':79,
'frisbee':30,
'umbrella':26,
'bottle':40,
'bus':6,
'bear':22,
'vase':76,
'toothbrush':80,
'spoon':45,
'train':7,
'sink':72,
'potted plant':59,
'handbag':27,
'cell phone':68,
'toaster':71,
'broccoli':51,
'refrigerator':73,
'laptop':64,
'remote':66,
'surfboard':38,
'cow':20,
'dining table':61,
'hot dog':53,
'car':3,
'sports ball':33,
'skateboard':37,
'dog':17,
'bed':60,
'cat':16,
'person':1,
'skis':31,
'giraffe':24,
'truck':8,
'parking meter':13,
'suitcase':29,
'cake':56,
'wine glass':41,
'baseball bat':35,
'backpack':25,
'zebra':23,
}

coco_trainid2cat = {value:key for key,value in coco_cat2trainid.items()}

# Classes in pascal dataset
PASCAL_CATS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car' , 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa',
               'train', 'tv/monitor']




COCO_PATH = '/data2/dataset/coco'


# Download Pascal VOC from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
PASCAL_PATH= '/data1/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012'
SBD_PATH = '/data1/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012'
image_size = (321, 321)


def get_cats(split, fold, num_folds=4):
    '''
      Returns a list of categories (for training/test) for a given fold number

      Inputs:
        split: specify train/val
        fold : fold number, out of num_folds
        num_folds: Split the set of image classes to how many folds. In BMVC paper, we use 4 folds

    '''
    num_cats = len(coco_cat2trainid.keys())
    assert(num_cats%num_folds==0)
    val_size = int(num_cats/num_folds)#80/4=20
    assert(fold<num_folds)
    val_set = [ fold*val_size+v for v in range(val_size)]
    train_set = [x for x in range(num_cats) if x not in val_set]
    if split=='train':
        return [coco_trainid2cat[x+1] for x in train_set]#start from 1
    else:
        return [coco_trainid2cat[x+1] for x in val_set]#start from 1


########################### The default setting ##########################################

empty_profile = Map(
                ###############################################
                # Do not change this part
                first_label_params=[('first_label', 1., 0.)],
                second_label_params=[('second_label', 1., 0.)],
                ###############################################
                k_shot=1,
                first_shape=None,
                second_shape=None,
                output_type=None,
                read_mode=None, # Either "Shuffle" (for training) or "Deterministic" (for testing, random seed fixed)
                scale_256=True,
                mean = (104.0/255, 116.0/255, 122.0/255),
                batch_size = 1,
                video_sets=[],
                image_sets=[],
                areaRng = [0 , np.inf],
                default_pascal_cats = PASCAL_CATS,
                default_coco_cats = None,
                pascal_cats = PASCAL_CATS,
                coco_cats = coco_cat2trainid.keys(),
                coco_path = COCO_PATH,
                pascal_path = PASCAL_PATH,
                sbd_path = SBD_PATH,
                worker_num = 4)


########################### Settings for reproducing experiments ###########################

"""
foldall_train = Map(empty_profile,
    read_mode='shuffle',
    image_sets=['sbd_training', 'pascal_training'],
    #image_sets=['pascal_training'],
    pascal_cats = PASCAL_CATS,
    first_shape=image_size,
    second_shape=image_size) # original code is second_shape=None),TODO

foldall_1shot_test = Map(empty_profile,
    db_cycle = 1000,
    read_mode='deterministic',
    image_sets=['pascal_test'],
    #image_sets=['pascal_training'],
    pascal_cats = PASCAL_CATS,
    first_shape=image_size,
    second_shape=image_size,
     k_shot=1 ) # original code is second_shape=None),TODO

foldall_5shot_test = Map(empty_profile,
    db_cycle = 1000,
    read_mode='deterministic',
    image_sets=['pascal_test'],
    #image_sets=['pascal_training'],
    pascal_cats = PASCAL_CATS,
    first_shape=image_size,
    second_shape=image_size,
     k_shot=5 ) # original code is second_shape=None),TODO

"""

#### fold 0 ####

# Setting for training (on **training images**)
fold0_train = Map(empty_profile,
    read_mode='shuffle',
    image_sets=['coco_train'],
    coco_cats = get_cats('train',0),
    first_shape=image_size,
    second_shape=image_size) # original code is second_shape=None),TODO

fold0_5shot_train = Map(fold0_train,k_shot=5)

# Setting for testing on **test images** in unseen image classes (in total 5 classes), 5-shot
fold0_5shot_test = Map(empty_profile,
    db_cycle = 1000,
    read_mode='deterministic',
    image_sets=['coco_train'],
    coco_cats = get_cats('test',0),
    first_shape=image_size,
    second_shape=image_size,
    k_shot=5)

## Setting for testing on **test images** in unseen image classes (in total 5 classes), 1-shot
fold0_1shot_test = Map(fold0_5shot_test, k_shot=1)
fold0_2shot_test = Map(fold0_5shot_test, k_shot=2)
fold0_3shot_test = Map(fold0_5shot_test, k_shot=3)
fold0_4shot_test = Map(fold0_5shot_test, k_shot=4)
fold0_5shot_test = Map(fold0_5shot_test, k_shot=5)
fold0_6shot_test = Map(fold0_5shot_test, k_shot=6)
fold0_7shot_test = Map(fold0_5shot_test, k_shot=7)
fold0_8shot_test = Map(fold0_5shot_test, k_shot=8)
fold0_9shot_test = Map(fold0_5shot_test, k_shot=9)
fold0_10shot_test = Map(fold0_5shot_test, k_shot=10)


#### fold 1 ####
fold1_train = Map(fold0_train, coco_cats=get_cats('train', 1))
fold1_5shot_train = Map(fold1_train,k_shot=5)
fold1_5shot_test = Map(fold0_5shot_test, coco_cats=get_cats('test', 1))
fold1_1shot_test = Map(fold1_5shot_test, k_shot=1)

#### fold 2 ####
fold2_train = Map(fold0_train, coco_cats=get_cats('train', 2))
fold2_5shot_train = Map(fold2_train,k_shot=5)
fold2_5shot_test = Map(fold0_5shot_test, coco_cats=get_cats('test', 2))
fold2_1shot_test = Map(fold2_5shot_test, k_shot=1)

#### fold 3 ####
fold3_train = Map(fold0_train, coco_cats=get_cats('train', 3))
fold3_5shot_train = Map(fold3_train,k_shot=5)
fold3_5shot_test = Map(fold0_5shot_test, coco_cats=get_cats('test', 3))
fold3_1shot_test = Map(fold3_5shot_test, k_shot=1)



if __name__ == '__main__':
    print(get_cats("train",0))