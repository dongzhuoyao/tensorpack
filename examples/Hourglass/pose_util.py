# Author: Tao Hu <taohu620@gmail.com>
import math
import numpy as np
from transforms import transform_preds

def final_preds(heatmap, coords, center, scale, res):
    # pose-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = heatmap[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if px > 1 and px < res[0] and py > 1 and py < res[1]:
                diff = np.array([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]])
                coords[n,p,0] += np.sign(diff).tolist()[0] * .25
                coords[n,p,1] += np.sign(diff).tolist()[0] * .25

    coords += 0.5
    preds = np.copy(coords)

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    return preds