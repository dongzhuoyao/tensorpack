# Author: Tao Hu <taohu620@gmail.com>
cimport cython
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.uint64_t, ndim=2] _fastUpdateConfusionMatrix(
    np.ndarray[np.uint16_t, ndim=2] pred,
    np.ndarray[np.uint16_t, ndim=2] label,
    np.ndarray[np.uint64_t, ndim=2] conf_m,
    const unsigned int nb_classes,
    const unsigned int ignore):

    cdef np.ndarray[np.uint16_t, ndim=1] flat_pred = np.ravel(pred)
    cdef np.ndarray[np.uint16_t, ndim=1] flat_label = np.ravel(label)

    for p, l in zip(flat_pred, flat_label):
        if l == ignore:
            continue
        if l < nb_classes and p < nb_classes:
            conf_m[l, p] += 1
        else:
            raise
    return conf_m

def fastUpdateConfusionMatrix(pred, label, conf_m, nb_classes, ignore):
    return _fastUpdateConfusionMatrix(pred, label, conf_m, nb_classes, ignore)