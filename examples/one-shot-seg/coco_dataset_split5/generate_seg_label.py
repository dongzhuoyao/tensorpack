# Author: Tao Hu <taohu620@gmail.com>
import numpy as np
from tensorpack.utils.segmentation.coco_util import generate_id2trainid, generate_image_mask,_build_vocab
import os,cv2
from tqdm import tqdm
#this file is used to generate segmentation label file for coco, for saving image loading times.


coco_img_path = "/data2/dataset/coco/train2014"
coco_seglabel_path = "/data2/dataset/coco/train2014_seg_label"

#os.mkdir(coco_seglabel_path)


def generate_mask(_coco,catId_to_ascendorder, img_id):
    img = _coco.loadImgs(img_id)[0]
    img_file_name = img['file_name']
    annIds = _coco.getAnnIds(imgIds=img_id)
    img_mask = np.zeros((img['height'], img['width'], 1), dtype=np.uint8)

    for annId in annIds:
        ann = _coco.loadAnns(annId)[0]

        img_mask, ann = generate_image_mask(_coco, img_mask, annId, catId_to_ascendorder)

    return img_file_name, img_mask


from pycocotools.coco import COCO
_coco = COCO("/data2/dataset/annotations/instances_train2014.json")
catId_to_ascendorder = generate_id2trainid(_coco)

print catId_to_ascendorder
for key,value in tqdm(_coco.imgs.items()):
    img_id = value['id']
    img_name = value['file_name']
    img_file_name, img_mask = generate_mask(_coco, catId_to_ascendorder, img_id=img_id)
    cv2.imwrite(os.path.join(coco_seglabel_path, img_name), img_mask)


print("catId_to_ascendorder length: {}".format(len(catId_to_ascendorder)))