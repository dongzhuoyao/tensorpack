#!/usr/bin/env bash

cur_basename=`basename "$0"`#https://stackoverflow.com/questions/192319/how-do-i-know-the-script-file-name-in-a-bash-script

result_txt = $cur_basename.".txt"

model_path = train_log/res101.slim.2branch.speedup:fold0_1shot_test/model-5563463

python res101.slim.2branch.speedup.py --test --gpu 4\
 --test_load $model_path\
  --test_data fold0_5shot_test >> $result_txt

python res101.slim.2branch.speedup.py --test --gpu 4\
 --test_load $model_path\
  --test_data fold0_5shot_test >> $result_txt

python res101.slim.2branch.speedup.py --test --gpu 4\
 --test_load $model_path\
  --test_data fold0_5shot_test >> $result_txt

python res101.slim.2branch.speedup.py --test --gpu 4\
 --test_load $model_path\
  --test_data fold0_5shot_test >> $result_txt