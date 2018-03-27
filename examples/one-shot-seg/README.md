

## Result 

Arch |Dataset| Result 
|----|----|----|
~~res101.slim|pascal0|53.8~~
~~res101.slim.360k.newlr||55.3~~
~~res101.slim.dp_new.lr1e-10|pascal0|22~~
~~res101.slim.2branch|pascal0|55.3~~
~~res101.slim.2branch.speedup.tailcbr:fold0_1shot_test|pascal0|43.4~~
~~res101.slim.2branch.speedup.taildilation_dl:fold0_1shot_test|pascal0|47.0~~
~~res101.slim.2branch.speedup.tailcbr.lr1e-5:fold0_1shot_test|pascal0|43.4~~
~~res101.slim.2branch.speedup.tailcbr.size473:fold0_1shot_test|pascal0|43.4~~
res101.slim.2branch.speedup|pascal0|49.9(backup), 51.4(backup2),47.5(backup3)
res101.slim.2branch.speedup.size473:fold0_1shot_test|pascal0|56.7
res101.slim.2branch.speedup.240k|pascal0|60
res101.slim.2branch.speedup.image473|pascal0|58.4
res101.slim.2branch.speedup.image473.lrschedule/model-7500|pascal0|58.8
res101.slim.2branch.speedup.size473.sbugfix|pascal0|44
res101.slim.2branch.speedup.mcontext.240k|pascal0|63.6
res101.slim.2branch.speedup.60kiter|pascal0|:question:
res101.slim.2branch.speedup.mcontext.240k.image473|pascal0|:question:
res101.slim.2branch.speedup.mcontext.240k.image473.lrschedule|pascal0|58(backup),58.4(backup2)
res101.slim.2branch.speedup.240k.forbash|pascal0|:question:

* experiments show that 1000 test and 300 test lead nearly no difference.
* image size 473 showes 3% lower than image size 321
* **important update:** 3.27, test data size changed to 1000 fixed


Arch |fold0| fold1|fold2|fold3|Mean 
|----|----|----|----|----|----|
|res101.slim.2branch.speedup.240k.forbash|59.7|53.8|50.7|47.9|53.0|
