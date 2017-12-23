#-*- coding: UTF-8 -*-
import os
import cv2,os
from tqdm import tqdm
import numpy as np
# cropRGBImage
def cropImage(filepath, outputpath, split_num):
    print "start crop images."
    pathDir = os.listdir(filepath)
    for filename in tqdm(pathDir):
        print filename
        child = os.path.join(filepath, filename)
        im = cv2.imread(child)
        lx,ly,lz = im.shape
        for i in range(0,split_num):
            for j in range(0,split_num):
                crop_im = im[i*lx/split_num:(i+1)*lx/split_num, j*ly/split_num:(j+1)*ly/split_num, :]
                a = os.path.basename(filename).strip(".tif")
                cv2.imwrite(os.path.join(outputpath,"{}_patch{}_{}.tif".format(a,i,j)), crop_im)

# cropBWImage
def cropBW(filepath, outputpath, split_num):
    print "start crop gt."
    pathDir = os.listdir(filepath)
    for filename in tqdm(pathDir):
        #if "yrol-w7" in filename:
            print filename
            child = os.path.join(filepath, filename)
            im = cv2.imread(child,0)
            lx,ly = im.shape
            for i in range(0,split_num):
                for j in range(0,split_num):
                    crop_im = im[i*lx/split_num:(i+1)*lx/split_num, j*ly/split_num:(j+1)*ly/split_num].astype(np.uint8)
                    crop_im = crop_im/255
                    #crop_im = crop_im[:,:,0]

                    print np.unique(crop_im)
                    a = os.path.basename(filename).strip(".tif")
                    cv2.imwrite(os.path.join(outputpath, "{}_patch{}_{}.tif".format(a, i, j)), crop_im)
                    tmp = cv2.imread(os.path.join(outputpath, "{}_patch{}_{}.tif".format(a, i, j)),0)
                    print np.unique(tmp)


def split_and_generate_txt(imagepath, gtpath, train_data_ratio):
    pathDir = os.listdir(imagepath)
    middle = int(train_data_ratio*len(pathDir))
    train_pathDir = pathDir[:middle]
    val_pathDir = pathDir[middle:]
    #train.txt
    f = file("train.txt", "w+")
    for filename in train_pathDir:
        child = os.path.join('{} {}\n'.format(os.path.join(imagepath, filename), os.path.join(gtpath, filename)))
        f.write(child)
    f.close()

    # val.txt
    f = file("val.txt", "w+")
    for filename in val_pathDir:
        child = os.path.join('{} {}\n'.format(os.path.join(imagepath, filename), os.path.join(gtpath, filename)))
        f.write(child)
    f.close()


def generate_test():
    testPath = "/data1/dataset/AerialImageDataset/test/images"
    targetPath = "/data1/dataset/jpg_aerial/final/test/"

    f = open("test.txt","w")
    import shutil
    #shutil.rmtree(targetPath)
    os.makedirs(targetPath)
    ll = os.listdir(testPath)
    for filename in tqdm(ll):
        img = cv2.imread(os.path.join(testPath,filename))
        true_name = filename.strip("_tif2jpg.jpg")
        cv2.imwrite(os.path.join(targetPath, "{}.jpg".format(true_name)),img)
        f.write(os.path.join(targetPath, "{}.jpg\n".format(true_name)))
    f.close()

def generate_train():
    imagePath = "/data1/dataset/AerialImageDataset/train/images"
    gtPath = "/data1/dataset/AerialImageDataset/train/gt"

    outputPath = "/data1/dataset/jpg_aerial/final"
    target_image_path = os.path.join(outputPath, "images")
    target_gt_path = os.path.join(outputPath, "gt")


    import shutil
    shutil.rmtree(outputPath)
    os.makedirs(target_image_path)
    os.makedirs(target_gt_path)

    # crop
    split_num = 5;  # how many patch in a row/column
    cropBW(gtPath, target_gt_path, split_num)
    cropImage(imagePath, target_image_path, split_num)


    # writeTxt
    split_and_generate_txt(target_image_path, target_gt_path, train_data_ratio=0.9)


if __name__ == '__main__':

    #generate_test()
    generate_train()