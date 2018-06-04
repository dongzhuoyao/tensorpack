
import numpy as np
import random,os
from multiprocessing import Process, Queue, Pool, Lock

import sys
import traceback
import util
from util import cprint, bcolors, DBImageItem
from skimage.transform import resize
import copy,cv2
from pycocotools.coco import COCO
from pycocotools import mask
from tensorpack.utils.segmentation.coco_util import generate_id2trainid, generate_image_mask
from tqdm import tqdm
from tensorpack.utils.segmentation.coco_util import catid2trainid, catid2catstr
coco_path = '/data2/dataset/coco/train2014'
coco_seglabel_path = "/data2/dataset/coco/train2014_seg_label"

is_debug = 0
images_per_class = 1000
area_limit = 1600

def generate_mask(coco, img_id, cat_id, is_read_image = True):
    img = coco.loadImgs(img_id)[0]
    img_file_name = img['file_name']
    img_mask = cv2.imread(os.path.join(coco_seglabel_path, img_file_name),cv2.IMREAD_GRAYSCALE)
    catId_to_ascendorder = generate_id2trainid(coco)
    train_id = catId_to_ascendorder[cat_id]
    img_mask_copy = np.copy(img_mask)
    img_mask[np.where(img_mask_copy == train_id)] = 1
    img_mask[np.where(img_mask_copy != train_id)] = 0

    if is_read_image:
        image = os.path.join(coco_path, "COCO_train2014_{}.jpg".format(str(img_id).zfill(12)))
        image = cv2.imread(image)
        return image, img_mask
    else:
        return img_mask


class DBCOCOItem(DBImageItem):
    def __init__(self, name, db_path, dataType, image_id, catId, coco_db, pycocotools):
        DBImageItem.__init__(self, name)
        self.image_id = image_id
        self.cat_id = catId
        self.db_path = db_path
        self.dataType = dataType
        self.coco_db = coco_db
        self.pycocotools = pycocotools
        self.img_path= os.path.join(coco_path,"COCO_train2014_{}.jpg".format(str(self.image_id).zfill(12)))
        self.mask_path = None

class COCO:
    def __init__(self, db_path, dataType):
        self.pycocotools = __import__('pycocotools.coco')
        if dataType == 'train':
            dataType = 'train2014'
        elif dataType == 'val':
            dataType = 'val2014'
        else:
            raise Exception('split \'' + dataType + '\' is not valid! Valid splits: training/test')

        self.db_path = db_path
        self.dataType = dataType
        annFile='%s/annotations/instances_%s.json' % (self.db_path, self.dataType)

        self.coco = self.pycocotools.coco.COCO(annFile)


    def getItems(self, cats=[], areaRng=[], iscrowd=False):

        catIds = self.coco.getCatIds(catNms=cats)
        clusters = {}

        import json
        result_file = "cluster.json-6.4"

        if is_debug==0 and os.path.isfile(result_file):
            cprint("recoverring from json file", bcolors.OKBLUE)
            with open(result_file,'r') as f:
                clusters_json = json.load(f)
                for catId in catIds:
                    clusters[str(catId)] = clusters_json[str(catId)]
                    cprint('Class:{}, totally {} items, filtered  items(whose area is smaller than {} pixels)'.format(
                        catid2catstr[catId], len(clusters[str(catId)]), area_limit), bcolors.OKBLUE)

        else:
            cprint("generating json file....", bcolors.OKBLUE)
            for catId in [value['id'] for key,value in self.coco.cats.items()]:
                image_ids = self.coco.getImgIds(catIds=[catId])
                filtered = 0
                for idx, image_id in tqdm(enumerate(image_ids), total=len(image_ids),desc="catId={}".format(catId)):
                    if is_debug==1 and idx>100:break
                    if idx>images_per_class:break # coco data is too large!!, use small scale data
                    item = {"image_id":image_id, "cat_id":catId,
                            "image_path": os.path.join(coco_path, "COCO_train2014_{}.jpg".format(str(image_id).zfill(12)))}
                    mask = generate_mask(self.coco, image_id, catId, is_read_image=False)
                    if mask.sum() < area_limit:# too small object
                        filtered += 1
                        continue

                    if not clusters.has_key(catId):
                        clusters[catId] = [item]
                    else:
                        clusters[catId].append(item)
                cprint('Class:{}, totally {} items, filtered {} items(whose area is smaller than {} pixels)'.format(catid2catstr[catId], len(clusters[catId]), filtered, area_limit), bcolors.OKBLUE)

            with open(result_file,"w") as f:
                json.dump(clusters, f)
                for key,value in clusters.items():
                    cprint(
                        'Class:{}, totally {} items after filtering items(whose area is smaller than {} pixels)'.format(
                            catid2catstr[key], len(value), area_limit), bcolors.OKBLUE)


        cprint('Total of ' + str(len(clusters)) + ' classes!', bcolors.OKBLUE)
        items = []
        for catId in catIds:
            items.extend([(clusters[str(catId)], idx) for idx,tmp  in  enumerate(clusters[str(catId)])])
        cprint(str(len(items)) + ' annotations read from coco', bcolors.OKGREEN)
        return items


class DBInterface():
    def __init__(self, params):
        self.params = params
        self.load_items()
        self.data_size = len(self.db_items)
        # initialize the random generator
        self.init_randget(params['read_mode'])
        self.cycle = 0

    def load_items(self):
        image_set = self.params['image_sets'][0]
        assert image_set.startswith('coco')
        coco_db = COCO(self.params['coco_path'], image_set.replace("coco_", ""))  # train or test
        self.coco = coco_db.coco
        self.db_items = coco_db.getItems(self.params['coco_cats'], self.params['areaRng'])
        cprint('Total of ' + str(len(self.db_items)) + ' coco db items loaded!', bcolors.OKBLUE)

        self.orig_db_items = copy.copy(self.db_items)
        self.seq_index = len(self.db_items)
        
    def init_randget(self, read_mode):
        self.rand_gen = random.Random()
        if read_mode == 'shuffle':
            self.rand_gen.seed()
        elif read_mode == 'deterministic':
            self.rand_gen.seed(1385) #>>>Do not change<<< Fixed seed for deterministic mode. 
    
    def update_seq_index(self):
        self.seq_index += 1
        if self.seq_index >= len(self.db_items):# reset status when full
            self.db_items = copy.copy(self.orig_db_items)
            self.rand_gen.shuffle(self.db_items)
            self.seq_index = 0
    
    def next_pair(self):
            end_of_cycle = self.params.has_key('db_cycle') and self.cycle >= self.params['db_cycle']
            if end_of_cycle:
                assert(self.params['db_cycle'] > 0) # full, reset status
                self.cycle = 0
                self.seq_index = len(self.db_items)
                self.init_randget(self.params['read_mode'])
                
            self.cycle += 1
            self.update_seq_index()

            imgset, second_index = self.db_items[self.seq_index] # query image index
            set_indices = range(second_index) + range(second_index+1, len(imgset)) # exclude second_index
            assert(len(set_indices) >= self.params['k_shot'])
            self.rand_gen.shuffle(set_indices)
            first_index = set_indices[:self.params['k_shot']] # support set image indexes(may be multi-shot~)

            metadata = {
                        'class_id': imgset[0]['cat_id'],
                        #'class_name':coco_trainid2cat[coco_cat2trainid[imgset[0].cat_id]],
                        'image1_name':[os.path.basename(imgset[ii]['image_path']) for ii in first_index],
                        'image2_name': os.path.basename(imgset[second_index]['image_path']),
                        }




            first_data = [generate_mask(self.coco, imgset[i]['image_id'], imgset[i]['cat_id']) for i in first_index]
            first_data_img = [d[0] for d in first_data]
            first_data_mask = [d[1] for d in first_data]

            second_data_img, second_data_mask = generate_mask(self.coco, imgset[second_index]['image_id'], imgset[second_index]['cat_id'])




            return [first_data_img,first_data_mask, \
                    second_data_img, second_data_mask,metadata ]








            

