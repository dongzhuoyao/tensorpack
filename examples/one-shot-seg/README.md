

## Result 

Arch |Dataset| Result 
|----|----|----|
~~|res101.slim|pascal0|53.8|~~
~~res101.slim.360k.newlr||55.3~~
~~res101.slim.dp_new.lr1e-10|pascal0|22~~
~~res101.slim.2branch|pascal0|55.3~~
~~res101.slim.2branch.speedup.tailcbr:fold0_1shot_test|pascal0|43.4~~
~~res101.slim.2branch.speedup.taildilation_dl:fold0_1shot_test|pascal0|47.0~~
~~res101.slim.2branch.speedup.tailcbr.lr1e-5:fold0_1shot_test|pascal0|43.4~~
~~res101.slim.2branch.speedup.tailcbr.size473:fold0_1shot_test|pascal0|43.4~~
res101.slim.2branch.speedup|pascal0|49.9(backup), 51.4(backup2),47.5(backup3)
res101.slim.2branch.speedup.size473(may be actually 321)|pascal0|56.7
res101.slim.2branch.speedup.image321|pascal0|54.7
res101.slim.2branch.speedup.image473|pascal0|58.4
res101.slim.2branch.speedup.image473.lrschedule|pascal0|58.8,58(backup),58.4(backup2)
res101.slim.2branch.speedup.size473.sbugfix|pascal0|44
res101.slim.2branch.speedup.240k|pascal0|60
res101.slim.2branch.speedup.mcontext.240k(256)|pascal0|63.6
res101.slim.2branch.speedup.mcontext.240k.lrschedule|pascal0|60.2
res101.slim.2branch.speedup.mcontext.240k.support_se|pascal0|63.5
res101.slim.2branch.speedup.mcontext.240k.all_se|pascal0|63.7
res101.slim.2branch.speedup.mcontext.240k.width64|pascal0|63.4
res101.slim.2branch.speedup.mcontext.240k.width128|pascal0|63.3
res101.slim.2branch.speedup.mcontext.240k.width384|pascal0|62.0
res101.slim.2branch.speedup.mcontext.240k.width512|pascal0|63.7
res101.slim.2branch.speedup.mcontext.240k.width1024|pascal0|63.2
res101.slim.2branch.speedup.mcontext.240k.image473.lrschedule|pascal0|59.3
res101.slim.2branch.speedup.mcontext.240k.image473|pascal0|60
res101.slim.2branch.speedup.mcontext3.240k|pascal0|61.2
res101.slim.2branch.speedup.mcontext23.240k|pascal0|63.2
res101.slim.2branch.speedup.mcontext123.240k|pascal0|61.4
res101.slim.2branch.speedup.60kiter|pascal0|61.5
res101.slim.2branch.speedup.60kiter.mcontext|pascal0|63.4
res101.slim.2branch.speedup.60kiter.mcontext.lrschedule|pascal0|61.8


* experiments show that 1000 test and 300 test lead nearly no difference.
* image size 473 showes 3% lower than image size 321
* **important update:** 3.27, test data size changed to 1000 fixed


Arch |n-shot|fold0| fold1|fold2|fold3|Mean 
|----|----|----|----|----|----|----|
|res101.slim.2branch.speedup.240k.forbash(346)|1-shot|59.7|53.8|50.7|47.9|53.0|
|res101.slim.2branch.speedup.240k.forbash(1000)|1-shot|59.0|53.7|50.4|47.9|53.0|
|res101.slim.2branch.speedup.240k.forbash(1000)|5-shot(prob mix)|58.8|53.4|50.5|48|52.6|
|res101.slim.2branch.speedup.240k.forbash(1000)|5-shot(or)|60.8|56.0|51.3|51.1|54.8|
|res101.slim.2branch.speedup.mcontext.240k|1-shot|62.4|55.7|51.8|51.1|55.3|
|res101.slim.2branch.speedup.mcontext.240k|5-shot|?|?|?|?|?|
|res101.slim.2branch.speedup.mcontext.240k.support_se|1-shot|63.3|54.3|51.2|51.5|55.0
|res101.slim.2branch.speedup.mcontext.240k.support_se|5-shot|?|55.6|?|53.6|?
|res101.slim.2branch.speedup.mcontext.240k.all_se|1-shot|63.7|53.8|50.6|?|stopped, no use
|res101.slim.2branch.speedup.mcontext.240k.width512|1-shot|63.4|55.4|52.2|52|55.75
|res101.slim.2branch.speedup.mcontext.240k.width512|5-shot|64.1|56.8|52.6|?|?|
|res101.slim.2branch.speedup.mcontext.240k.all_ran|1-shot|
|res101.slim.2branch.speedup.mcontext.240k.support_ran|1-shot|



oracle all class:  
res101.slim.2branch.speedup.mcontext.240k.width512.oracle:foldall_1shot_test, 67.0

