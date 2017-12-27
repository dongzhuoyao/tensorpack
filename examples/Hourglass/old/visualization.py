import argparse

from config import config
from hourglass import *
from dataset import preprocessing
import numpy as np
import cv2
import math
import sys
#import ipdb
#sys.path.insert(0, '../../../data/COCO/')
#from COCOAllJoints import COCOJoints


colors = np.random.randint( 0, 256, (config.nr_skeleton, 3) )


def paint_pixel( img, x, y, c, ratio ):
    img[ x, y, 0 ] = int( c[ 0 ] * ratio )
    img[ x, y, 1 ] = int( c[ 1 ] * ratio )
    img[ x, y, 2 ] = int( c[ 2 ] * ratio )

def draw_skeleton(aa, sk):
    for j in range(config.nr_skeleton):
        cv2.circle( aa, tuple(sk[j]), 2, tuple((255,0,0)), 2 )
    cv2.line( aa, tuple(sk[0]), tuple(sk[1]), (0,255,255), 2 )
    cv2.line( aa, tuple(sk[1]), tuple(sk[2]), (0,255,255), 2 )
    cv2.line( aa, tuple(sk[2]), tuple(sk[6]), (0,255,255), 2 )
    cv2.line( aa, tuple(sk[6]), tuple(sk[3]), (0,255,255), 2 )
    cv2.line( aa, tuple(sk[3]), tuple(sk[4]), (0,255,255), 2 )
    cv2.line( aa, tuple(sk[4]), tuple(sk[5]), (0,255,255), 2 )
    cv2.line( aa, tuple(sk[6]), tuple(sk[7]), (0,255,255), 2 )
    cv2.line( aa, tuple(sk[7]), tuple(sk[8]), (0,255,255), 2 )
    cv2.line( aa, tuple(sk[8]), tuple(sk[9]), (0,255,255), 2 )
    cv2.line( aa, tuple(sk[8]), tuple(sk[12]), (0,255,255), 2 )
    cv2.line( aa, tuple(sk[12]), tuple(sk[11]), (0,255,255), 2 )
    cv2.line( aa, tuple(sk[11]), tuple(sk[10]), (0,255,255), 2 )
    cv2.line( aa, tuple(sk[8]), tuple(sk[13]), (0,255,255), 2 )
    cv2.line( aa, tuple(sk[13]), tuple(sk[14]), (0,255,255), 2 )
    cv2.line( aa, tuple(sk[14]), tuple(sk[15]), (0,255,255), 2 )

def draw_skeleton_new(canvas,sk):
    #stickwidth = 3
    stickwidth = 7
    limbSeq = [
        [0,1],
        [1,2],
        [2,6],
        [6,3],
        [3,4],
        [4,5],
        [6,7],
        [7,8],
        [8,9],
        [8,12],
        [12,11],
        [11,10],
        [8,13],
        [13,14],
        [14,15]
    ]
    # visualize
    colors = [ [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255]]
    # subset is all instances of each image
    # candidate is all points' info of each image ,include(x,y,point_score,point_id)
    cur_canvas = canvas.copy()
    for i in range(len(limbSeq)):
        #for n in range(len(subset)):
            index = np.array(limbSeq[i])
            if sk[index[0],0] == -1 or sk[index[1],0] == -1:
                 continue
            Y = sk[index.astype(int), 0]
            X = sk[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])

    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    for j in range(config.nr_skeleton):
        #cv2.circle(canvas, tuple(sk[j]), 2, tuple((255, 0, 0)), 2)
        cv2.circle( canvas, tuple(sk[j]), 6, tuple((255,0,0)), 2 )

    return canvas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='model_path', help='path of model to test')
    args = parser.parse_args()

    output_num = 1
    func = load_model(args.model_path);
    #d = COCOJoints()
    d = MegviiRecordJoints('8781_megvii_record')
    train_data, test_data = d.load_data()

    color = np.random.randint(0, 255, (config.nr_skeleton, 3))
    batch_size = config.batch_size
    for test_id in range(0, len(test_data), batch_size):
        start_id = test_id
        end_id = min(len(test_data), test_id+batch_size)

        test_img, test_label = preprocessing(test_data[start_id:end_id], shape=config.data_shape, return_headRect=False, stage='val')

        res = func( data=test_img )
        res = np.array(res[0])

        for test_image_id in range(0, end_id-start_id):
            img = test_img[test_image_id].transpose(1, 2, 0)
            img = img*255+config.pixel_means
            cv2.imshow( "ori", img.astype(np.uint8) )
            com = np.zeros(img.shape, dtype=np.uint8); com[:,:,:] = img[:,:,:];
            htm = np.zeros((config.data_shape[0], config.data_shape[1], 3), dtype=np.uint8)
            r0 = res[ test_image_id ].copy();
            r0 = cv2.GaussianBlur( r0, (9, 9), 0 ); r0 /= 255;
            for w in range(config.nr_skeleton):
                res[ test_image_id, w ] /= np.amax( res[ test_image_id, w ] )
            for x in range( config.output_shape[1] ):
                for y in range( config.output_shape[0] ):
                    mx = 0; mk = 0
                    for w in range( config.nr_skeleton ):
                        try:
                            #if res[ test_image_id, w, x, y ] > mx:
                            #    mx = res[ test_image_id, w, x, y ]
                            if res[ test_image_id, w, y, x ] > mx:
                                mx = res[ test_image_id, w, y, x ]
                                mk = w
                        except:
                            from IPython import embed; embed()
                    for xx in range( x*4, x*4+4 ):
                        for yy in range( y*4, y*4+4 ):
                            #paint_pixel( htm, xx, yy, color[ mk ], mx )
                            paint_pixel( htm, yy, xx,  color[ mk ], mx )
            sk = [ (0, 0, 0) for i in range( config.nr_skeleton ) ]
            cov = [ 0 for i in range( config.nr_skeleton ) ]
            border = 0
            #dr = np.zeros((config.nr_skeleton, config.output_shape[0] + 2*border, config.output_shape[1] + 2*border))
            #dr[:, border:-border, border:-border] = res[ test_image_id ].copy()
            dr = res[test_image_id].copy()
            for w in range(config.nr_skeleton):
                dr[ w ] = cv2.GaussianBlur( dr[ w ], (9, 9), 0 )
            score = 0;
            for w in range( config.nr_skeleton ):
                lb = dr[ w ].argmax()
                x, y = np.unravel_index( lb, dr[ w ].shape )
                x -= border; y -= border;
                try:
                    score += r0[ w, x, y ];
                except:
                    from IPython import embed; embed()
                sk[ w ] = (y*4+2, x*4+2, r0[w, x, y])
            print(score);
            print(sk)
            sk = np.array(sk)
            draw_skeleton( htm, sk )
            draw_skeleton( com, sk );
            cv2.imshow( "com", com );
            cv2.imshow( "res", htm )
            key = cv2.waitKey()

if __name__ == '__main__':
    main()
