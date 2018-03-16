# Author: Tao Hu <taohu620@gmail.com>

import glob
# Author: Tao Hu <taohu620@gmail.com>
train_images = "/data1/dataset/m1-mar16-55-cropped/src"
train_gt = "/data1/dataset/m1-mar16-55-cropped/gt"


import shutil,os,cv2
from tqdm import tqdm
import numpy as np

#os.makedirs(expand_train_gt)
#os.makedirs(expand_val_gt)

print "train gt.."
for filename in tqdm(os.listdir(train_gt)):
    origin_img = cv2.imread(os.path.join(train_gt,filename),0)
    img = origin_img*255
    edge = cv2.Canny(img, 100, 200)
    pos = np.where(edge==255)
    cv2.imwrite(os.path.join(expand_train_gt,filename),edge)





