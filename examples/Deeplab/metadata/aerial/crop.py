#-*- coding: UTF-8 -*-
import os
import cv2,os
from tqdm import tqdm

# cropRGBImage
def cropImage(filepath, outputpath, split_num):
    print "start crop images."
    pathDir = os.listdir(filepath)
    for filename in tqdm(pathDir):
        print filename
        child = os.path.join('%s%s' % (filepath, filename))
        im = cv2.imread(child)
        lx,ly,lz = im.shape
        for i in range(0,split_num):
            for j in range(0,split_num):
                crop_im = im[i*lx/split_num:(i+1)*lx/split_num, j*ly/split_num:(j+1)*ly/split_num, :]
                a = os.path.basename(filename).strip(".jpg")
                cv2.imwrite(os.path.join(outputpath,"{}_patch{}_{}.jpg".format(a,i,j)), crop_im)

# cropBWImage
def cropBW(filepath, outputpath, split_num):
    print "start crop gt."
    pathDir = os.listdir(filepath)
    for filename in tqdm(pathDir):
        print filename
        child = os.path.join('%s%s' % (filepath, filename))
        im = cv2.imread(child)
        lx,ly,_ = im.shape
        for i in range(0,split_num):
            for j in range(0,split_num):
                crop_im = im[i*lx/split_num:(i+1)*lx/split_num, j*ly/split_num:(j+1)*ly/split_num]
                crop_im = crop_im/255
                crop_im = crop_im[:,:,0]
                a = os.path.basename(filename).strip(".jpg")
                cv2.imwrite(os.path.join(outputpath, "{}_patch{}_{}.jpg".format(a, i, j)), crop_im)


def split_and_generate_txt(imagepath, gtpath, train_data_ratio):
    pathDir = os.listdir(imagepath)
    middle = int(train_data_ratio*len(pathDir))
    train_pathDir = pathDir[:middle]
    val_pathDir = pathDir[middle:]
    #train.txt
    f = file("train.txt", "w+")
    for filename in train_pathDir:
        child = os.path.join('%s%s %s%s\n' % (imagepath, filename, gtpath, filename))
        f.write(child)
    f.close()

    # val.txt
    f = file("val.txt", "w+")
    for filename in val_pathDir:
        child = os.path.join('%s%s %s%s\n' % (imagepath, filename, gtpath, filename))
        f.write(child)
    f.close()


def generate_test():
    testPath = "/data1/dataset/jpg_aerial/test/images"
    targetPath = "/data1/dataset/jpg_aerial/final/test/"

    f = open("test.txt","w")
    import shutil
    shutil.rmtree(targetPath)
    os.makedirs(targetPath)
    ll = os.listdir(testPath)
    for filename in tqdm(ll):
        img = cv2.imread(os.path.join(testPath,filename))
        true_name = filename.strip("_tif2jpg.jpg")
        cv2.imwrite(os.path.join(targetPath, "{}.jpg".format(true_name)),img)
        f.write(os.path.join(targetPath, "{}.jpg\n".format(true_name)))
    f.close()

def generate_train():
    imagePath = "/data1/dataset/jpg_aerial/train/images/"
    gtPath = "/data1/dataset/jpg_aerial/train/gt/"

    outputPath = "/data1/dataset/jpg_aerial/final"
    """
    import shutil
    shutil.rmtree(outputPath)
    os.makedirs(os.path.join(outputPath, "images"))
    os.makedirs(os.path.join(outputPath, "gt"))

    # crop
    split_num = 5;  # how many patch in a row/column
    cropBW(gtPath, os.path.join(outputPath, "gt"), split_num)
    cropImage(imagePath, os.path.join(outputPath, "images"), split_num)
    """
    # writeTxt
    split_and_generate_txt(os.path.join(outputPath, "images"), os.path.join(outputPath, "gt"), train_data_ratio=0.9)


if __name__ == '__main__':

    #generate_test()
    generate_train()