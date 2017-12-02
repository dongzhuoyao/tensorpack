# Author: Tao Hu <taohu620@gmail.com>
import cv2

src = "/data_a/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg"
img = cv2.imread(src)
edge = cv2.Canny(img, 100, 200)
edge = cv2.GaussianBlur(edge,(5,5),0)
cv2.imwrite("edge.jpg",edge)
