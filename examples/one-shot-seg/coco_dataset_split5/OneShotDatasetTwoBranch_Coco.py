# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2

from tensorpack.utils import logger
from tensorpack.dataflow.base import RNGDataFlow
import  coco_datalayer, coco_settings
from tensorpack.utils.segmentation.coco_util import catid2trainid, catid2catstr

__all__ = ['OneShotDatasetTwoBranchCoco']

is_debug = 0

class OneShotDatasetTwoBranchCoco(RNGDataFlow):
    def __init__(self,name, image_size=(321,321)):
        self.name = name
        profile = getattr(coco_settings, name)
        profile_copy = profile.copy()
        profile_copy['deploy_mode'] = True
        profile_copy['first_shape'] = image_size
        profile_copy['second_shape'] = image_size
        self.dbi = coco_datalayer.DBInterface(profile)
        self.image_size = image_size


        self.data_size = len(self.dbi.db_items)
        if "test" in self.name:
            self.data_size = 1000

    def size(self):
        if is_debug == 1:
            return 50
        if "test" in self.name:
            return 1000
        else:
            return self.data_size

    @staticmethod
    def class_num():
        return 2

    def get_data(self): # only for one-shot learning
        for i in range(self.data_size):
            first_image_list,first_label_list,second_image, second_label, metadata = self.dbi.next_pair()
            #print metadata['class_id']
            #print catid2catstr[metadata['class_id']]
            if "train" in self.name:
                k_shots = len(first_image_list)
                class_id = metadata['class_id']
                first_image_masks = []
                for kk in range(k_shots):
                    first_image = first_image_list[kk]
                    first_label = first_label_list[kk]
                    first_image = cv2.resize(first_image, self.image_size)
                    first_label = cv2.resize(first_label, self.image_size, interpolation=cv2.INTER_NEAREST)
                    first_image_masked = first_image * first_label[:, :, np.newaxis]
                    first_image_masks.append(first_image_masked)


                second_image = cv2.resize(second_image, self.image_size)
                second_label = cv2.resize(second_label, self.image_size, interpolation=cv2.INTER_NEAREST)
                yield [np.stack(first_image_masks), second_image, second_label]
            else:
                k_shots = len(first_image_list)
                class_id = metadata['class_id']
                first_image_masks = []
                for kk in range(k_shots):
                    first_image = first_image_list[kk]
                    first_label = first_label_list[kk]
                    first_image = cv2.resize(first_image, self.image_size)
                    first_label = cv2.resize(first_label, self.image_size, interpolation=cv2.INTER_NEAREST)
                    first_image_masked = first_image * first_label[:, :, np.newaxis]
                    first_image_masks.append(first_image_masked)

                # second_image = cv2.resize(second_image, support_image_size)
                # second_label = cv2.resize(second_label, support_image_size, interpolation=cv2.INTER_NEAREST)
                yield [np.stack(first_image_masks), second_image, second_label]





if __name__ == '__main__':
    ds = OneShotDatasetTwoBranchCoco("fold0_5shot_train")
    from tensorpack.utils.segmentation.segmentation import visualize_label
    for idx,data in enumerate(ds.get_data()):
        first_img_masks, second_img, second_mask = data
        cv2.imwrite("first_image_mask.jpg", first_img_masks[0])
        cv2.imwrite("second_image.jpg", second_img)
        cv2.imwrite("second_mask.jpg", visualize_label(second_mask))
        pass
