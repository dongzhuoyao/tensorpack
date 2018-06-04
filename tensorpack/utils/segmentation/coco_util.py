# Author: Tao Hu <taohu620@gmail.com>
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask
from collections import Counter

catid2trainid = {
1:1,
2:2,
3:3,
4:4,
5:5,
6:6,
7:7,
8:8,
9:9,
10:10,
11:11,
13:12,
14:13,
15:14,
16:15,
17:16,
18:17,
19:18,
20:19,
21:20,
22:21,
23:22,
24:23,
25:24,
27:25,
28:26,
31:27,
32:28,
33:29,
34:30,
35:31,
36:32,
37:33,
38:34,
39:35,
40:36,
41:37,
42:38,
43:39,
44:40,
46:41,
47:42,
48:43,
49:44,
50:45,
51:46,
52:47,
53:48,
54:49,
55:50,
56:51,
57:52,
58:53,
59:54,
60:55,
61:56,
62:57,
63:58,
64:59,
65:60,
67:61,
70:62,
72:63,
73:64,
74:65,
75:66,
76:67,
77:68,
78:69,
79:70,
80:71,
81:72,
82:73,
84:74,
85:75,
86:76,
87:77,
88:78,
89:79,
90:80,
}


catstr2trainid = {
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

trainid2catstr = {value:key for key,value in catstr2trainid.items()}

catid2catstr = {catid: trainid2catstr[catid2trainid[catid]] for catid in  catid2trainid.keys()}



def _build_vocab(caption_list, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(caption_list):
        words = caption.split(' ')  # caption contrains only lower-case words
        for w in words:
            counter[w] += 1

        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx

def generate_id2trainid(_coco):
    new_dict = {}
    for idx,data in enumerate(_coco.cats.items()):
        key,value = data #start from 1!!!
        new_dict[key] = idx+1

    return new_dict


def generate_image_mask(_coco, img_mask, annId, cat_dict):
    height,width,_ =img_mask.shape
    ann = _coco.loadAnns(annId)[0]

    # polygon
    if type(ann['segmentation']) == list:
        for _instance in ann['segmentation']:
            rle = mask.frPyObjects([_instance], height, width)
            m = mask.decode(rle)
            img_mask[np.where(m == 1)] = cat_dict[ann['category_id']]
    # mask
    else:  # mostly is aeroplane
        if type(ann['segmentation']['counts']) == list:
            rle = mask.frPyObjects([ann['segmentation']], height, width)
        else:
            rle = [ann['segmentation']]
        m = mask.decode(rle)
        img_mask[np.where(m == 1)] = cat_dict[ann['category_id']]



    return img_mask, ann

