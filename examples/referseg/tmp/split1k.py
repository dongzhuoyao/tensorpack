# Author: Tao Hu <taohu620@gmail.com>


import os
from tqdm import tqdm
from shutil import copyfile


#target_path = "/Users/ht/Desktop/new_train2014_assigned"
#src_path = "/Users/ht/Desktop/new_train2014"
target_path = "/Users/ht/Desktop/new_val2014_assigned"
src_path = "/Users/ht/Desktop/new_val2014"
os.mkdir(target_path)



ll = os.listdir(src_path)
ll.sort()
mylist = ll[:10000]
for i in range(1):
    print i
    cur_dir = os.path.join(target_path,str(i))
    os.mkdir(cur_dir)
    for filename in tqdm(mylist[i*1000:i*1000+1000]):
        copyfile(os.path.join(src_path, filename),os.path.join(cur_dir, filename))






