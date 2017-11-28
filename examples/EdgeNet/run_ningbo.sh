#!/usr/bin/env bash
python edgenet.py --data_dir /data_a/dataset/ningbo3539 \
 --meta_dir ningbo --gpu 2 \
  --edge_dir /data_a/dataset/ningbo3539_edge_gt \
--crop_size 256 \
--val_crop_size 256