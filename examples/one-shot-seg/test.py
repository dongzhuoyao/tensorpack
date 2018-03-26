"""
test.py - Test network using low-shot pairs
"""

# TODO: Put all the copied files in a temp folder and operate out of there. Or no copy, just shift to the fcn dir
# current system is bad
import os
import sys
import numpy as np
import ss_datalayer
import csv
from skimage.io import imsave
import time

# Get image pairs
class LoaderOfPairs(object):
    def __init__(self, profile):
        profile_copy = profile.copy()
        profile_copy['first_label_params'].append(('original_first_label', 1.0, 0.0))
        profile_copy['deploy_mode'] = True
        dbi = ss_datalayer.DBInterface(profile)
        self.data_size = len(dbi.db_items)
        self.PLP = ss_datalayer.PairLoaderProcess(None, None, dbi, profile_copy)
    def get_items(self):
        self.out = self.PLP.load_next_frame(try_mode=False)
        return (np.asarray(self.out['first_img']), #[np.newaxis,:,:,:], 
                np.asarray(self.out['first_label']), #[np.newaxis,:,:,:], 
                np.asarray(self.out['second_img']), #[np.newaxis,:,:,:], 
                np.asarray(self.out['second_label']), #[np.newaxis,:,:,:], 
                self.out['deploy_info'])
    def get_items_no_return(self):
        self.out = self.PLP.load_next_frame(try_mode=False)





