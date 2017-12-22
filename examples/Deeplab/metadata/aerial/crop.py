#-*- coding: UTF-8 -*-
import os
import scipy.misc as misc

# cropRGBImage
def cropImage(filepath, outputpath, split_num):
    pathDir = os.listdir(filepath)
    for filename in pathDir:
        child = os.path.join('%s%s' % (filepath, filename))
        im = misc.imread(child)
        lx,ly,lz = im.shape
        for i in range(0,split_num):
            for j in range(0,split_num):
                crop_im = im[i*lx/split_num:(i+1)*lx/split_num, j*ly/split_num:(j+1)*ly/split_num, :]
                a = filename
                a.replace('.tif', '')
                misc.imsave(outputpath + a + 'patch'+str(i)+'_'+str(j)+'.jpg', crop_im)
        #print child.decode('gbk')  # .decode('gbk')是解决中文显示乱码问题

# cropBWImage
def cropBW(filepath, outputpath, split_num):
    pathDir = os.listdir(filepath)
    for filename in pathDir:
        child = os.path.join('%s%s' % (filepath, filename))
        im = misc.imread(child)
        lx,ly = im.shape
        for i in range(0,split_num):
            for j in range(0,split_num):
                crop_im = im[i*lx/split_num:(i+1)*lx/split_num, j*ly/split_num:(j+1)*ly/split_num]
                a = filename
                a.replace('.tif', '')
                misc.imsave(outputpath + a + 'patch'+str(i)+'_'+str(j)+'.jpg', crop_im)
        #print child.decode('gbk')  # .decode('gbk')是解决中文显示乱码问题


def writeTxt(imagepath,gtpath,outputName):
    pathDir = os.listdir(imagepath)
    f = file(outputName, "w+")
    for filename in pathDir:
        child = os.path.join('%s%s %s%s\n' % (imagepath, filename, gtpath, filename))
        f.write(child)
    f.close()


# 读取文件内容并打印
def readFile(filename):
    fopen = open(filename, 'r')  # r 代表read
    for eachLine in fopen:
        print "读取到得内容如下：", eachLine
    fopen.close()


# 输入多行文字，写入指定文件并保存到指定文件夹
def writeFile(filename):
    fopen = open(filename, 'w')
    print "\r请任意输入多行文字", " ( 输入 .号回车保存)"
    while True:
        aLine = raw_input()
        if aLine != ".":
            fopen.write('%s%s' % (aLine, os.linesep))
        else:
            print "文件已保存!"
            break
    fopen.close()


if __name__ == '__main__':
    imagePath = "/data_a/dataset/AerialImageCroppedDataset/train/images/"
    gtPath = "/data_a/dataset/AerialImageCroppedDataset/train/gt/"
    outputPath = "~/"

    #crop
    #split_num = 5;#how many patch in a row/column
    #cropImage(imagePath, outputPath, split_num)
    #cropBW(gtPath, outputPath, split_num)

    #writeTxt
    writeTxt(imagePath, gtPath, "train.txt")

    #filePath = "D:\\FileDemo\\Java\\myJava.txt"
    #filePathI = "D:\\FileDemo\\Python\\pt.py"
    #readFile(filePath)
    #writeFile(filePathI)
