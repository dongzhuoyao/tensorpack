import cv2


def draw_skeleton(input_image, sk):
    for j in range(sk.shape[0]):
        cv2.circle(input_image, tuple(sk[j]), 2, tuple((255, 0, 0)), 2)
    cv2.line(input_image, tuple(sk[0]), tuple(sk[1]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[1]), tuple(sk[2]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[2]), tuple(sk[6]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[6]), tuple(sk[3]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[3]), tuple(sk[4]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[4]), tuple(sk[5]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[6]), tuple(sk[7]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[7]), tuple(sk[8]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[8]), tuple(sk[9]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[8]), tuple(sk[12]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[12]), tuple(sk[11]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[11]), tuple(sk[10]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[8]), tuple(sk[13]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[13]), tuple(sk[14]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[14]), tuple(sk[15]), (0, 255, 255), 2)




def visualize(oriImg, points, pa):
    import matplotlib
    import cv2 as cv
    import matplotlib.pyplot as plt
    import math

    fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(12, 12)

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170,0,255],[255,0,255]]
    canvas = oriImg
    stickwidth = 4
    x = points[:, 0]
    y = points[:, 1]

    for n in range(len(x)):
        for child in range(len(pa)):
            if pa[child] is 0:
                continue

            x1 = x[pa[child] - 1]
            y1 = y[pa[child] - 1]
            x2 = x[child]
            y2 = y[child]

            cv.line(canvas, (x1, y1), (x2, y2), colors[child], 8)


    plt.imshow(canvas[:, :, [2, 1, 0]])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 12)

    from time import gmtime, strftime
    import os
    directory = 'data/mpii/result/test_images'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fn = os.path.join(directory, strftime("%Y-%m-%d-%H_%M_%S", gmtime()) + '.jpg')

    plt.savefig(fn)