#!/usr/bin/env bash
python unet.py --data_dir /data_a/dataset/ningbo4763 \
 --meta_dir ningbo4763 --gpu 3 \
--crop_size 512 \
--val_crop_size 512 \
--batch_size 8