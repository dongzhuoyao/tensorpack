#!/bin/sh


python res101.slim.2branch.speedup.mcontext.240k.py --gpu 3\
  --train_data fold0_train\
  --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.width512:fold0_1shot_test/model-5582628\
  --test_data fold0_1shot_test >> res101.slim.2branch.speedup.mcontext.240k.batch.result

python res101.slim.2branch.speedup.mcontext.240k.py --gpu 3\
  --train_data fold1_train\
  --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.width512:fold1_1shot_test/model-5579295\
  --test_data fold1_1shot_test >> res101.slim.2branch.speedup.mcontext.240k.batch.result

python res101.slim.2branch.speedup.mcontext.240k.py --gpu 3\
  --train_data fold2_train\
  --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.width512:fold2_1shot_test/model-5575962\
  --test_data fold2_1shot_test >> res101.slim.2branch.speedup.mcontext.240k.batch.result

python res101.slim.2branch.speedup.mcontext.240k.py --gpu 3\
  --train_data fold3_train\
  --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.width512:fold3_1shot_test/model-5582628\
  --test_data fold3_1shot_test >> res101.slim.2branch.speedup.mcontext.240k.batch.result