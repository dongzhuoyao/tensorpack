#!/usr/bin/env bash
python edgenet.3gpu.py --data_dir /data_a/dataset/ningbo3539 \
 --meta_dir ningbo --gpu 1,2,3 \
  --edge_dir /data_a/dataset/ningbo3539_edge_gt \
--crop_size 256 \
--val_crop_size 256