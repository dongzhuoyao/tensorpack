# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2
from tqdm import tqdm

from tensorpack.utils import logger
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.utils.segmentation.segmentation import visualize_label
__all__ = ['DataLoader']

# maximum length of caption(number of word). if caption is longer than max_length, deleted.
max_length = 49
# if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
word_count_threshold = 1



from collections import Counter
import numpy as np
import pandas as pd
import os
import json
from pycocotools.coco import COCO
from pycocotools import mask

def _process_caption_data(caption_file, image_dir, max_length):
    coco = COCO(caption_file)
    caption_data = coco.dataset
    img_ids = [tmp['id'] for tmp in caption_data["images"]]

    # id_to_filename is a dictionary such as {image_id: filename]}
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
    data = []
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id])
        data += [annotation]

    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data) # one image corresponds to 5 captions
    del caption_data['id']
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)

    del_idx = []
    for i, caption in tqdm(enumerate(caption_data['caption'])):
        caption = caption.replace('.', '').replace(',', '').replace("'", "").replace('"', '')
        caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
        caption = " ".join(caption.split())  # replace multiple spaces

        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)

    # delete captions if size is larger than max_length
    print "The number of captions before deletion: %d" % len(caption_data)
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print "The number of captions after deletion: %d" % len(caption_data)

    #recover back to dict by dongzhuoyao
    img_dict = {}
    for row in tqdm(caption_data.itertuples()):
        img_dict[row.image_id] = row.caption

    return caption_data, img_dict, coco


def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
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


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples, max_length + 2)).astype(np.int32)

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ")  # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])

        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>'])

        captions[i, :] = np.asarray(cap_vec)
    print "Finished building caption vectors"
    return captions


def generate_mask(_coco, img_id):
    img = _coco.loadImgs(img_id)[0]
    img_file_name = img['file_name']
    annIds = _coco.getAnnIds(imgIds=img_id)
    img_mask = np.zeros((img['height'], img['width'], 1), dtype=np.uint8)

    for annId in annIds:
        ann = _coco.loadAnns(annId)[0]

        # polygon
        if type(ann['segmentation']) == list:
            for _instance in ann['segmentation']:
                rle = mask.frPyObjects([_instance], img['height'], img['width'])
        # mask
        else:  # mostly is aeroplane
            if type(ann['segmentation']['counts']) == list:
                rle = mask.frPyObjects([ann['segmentation']], img['height'], img['width'])
            else:
                rle = [ann['segmentation']]
        m = mask.decode(rle)
        img_mask[np.where(m == 1)] = ann['category_id']

    return img_file_name, img_mask


def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs

class DataLoader(RNGDataFlow):
    def __init__(self,name, img_dir, train_img_num = 4000, caption_num_per_image = 1):
        if "train" in name:
            self.image_dir = img_dir
            # about 80000 images and 400000 captions for train dataset
            panda_caption, img_dict, coco_caption = _process_caption_data(caption_file='/data2/dataset/annotations/captions_train2014.json',
                                                  image_dir=img_dir,
                                                  max_length=max_length)

            word_to_idx = _build_vocab(annotations=panda_caption, threshold=word_count_threshold)
            print("build vocab done...")
            self.word_to_idx = word_to_idx
            #with open("tmp.json","w") as f:
            #    json.dump(self.img_dict,f)

            self.img_ids = img_dict.keys()
            self.img_dict = img_dict
            self.coco_caption = coco_caption
            self.coco_instance = COCO("/data2/dataset/annotations/instances_train2014.json")


        elif "test" in name:
            panda_caption, img_dict, coco_caption = _process_caption_data(caption_file='/data2/dataset/annotations/captions_val2014.json',
                                                image_dir=img_dir,
                                                max_length=max_length)
            word_to_idx = _build_vocab(annotations=panda_caption, threshold=word_count_threshold)
            print("build vocab done...")
            self.word_to_idx = word_to_idx
            self.img_dict = img_dict
            self.img_ids = img_dict.keys()
            self.coco_caption = coco_caption
            self.coco_instance = COCO("/data2/dataset/annotations/instances_val2014.json")
            self.image_dir = img_dir

        else:
            raise


    def size(self):
        return len(self.img_dict.keys())

    @staticmethod
    def class_num():
        return 80 #Coco

    def get_data(self): # only for one-shot learning
        for i in range(self.size()):
            img_id = self.img_ids[i]
            caption = self.img_dict[img_id]# only consider one caption
            img_file_name, gt = generate_mask(self.coco_instance,img_id)
            img = cv2.imread(os.path.join(self.image_dir,img_file_name))
            yield [img, gt, caption]



if __name__ == '__main__':
    ds = DataLoader("test","/data2/dataset/coco/val2014")
    for idx, data in enumerate(ds.get_data()):
        img, gt, caption = data[0],data[1],data[2]
        print("{}".format(caption))
        cv2.imshow("img", img)
        cv2.imshow("label", visualize_label(gt,class_num=80))
        cv2.waitKey(50000)


