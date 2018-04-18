# Author: Tao Hu <taohu620@gmail.com>
# the main code is borrowed from https://github.com/yunjey/show-attend-and-tell/blob/master/prepro.py
from collections import Counter
from pycocotools.coco import COCO

def _build_vocab(annotations, word_count_threshold=1):
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations):
        words = caption.split(' ')  # caption contrains only lower-case words
        for w in words:
            counter[w] += 1

        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= word_count_threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), word_count_threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx

caption_json_train = "/data2/dataset/annotations/captions_train2014.json"
caption_json_val = "/data2/dataset/annotations/captions_val2014.json"

coco_caps = COCO(caption_json_val)
coco_vocabulary = "coco_vocabulary.txt"

anns = coco_caps.anns
anns = anns.values()
captions = [a['caption'] for a in anns]
result = _build_vocab(captions)

with open(coco_vocabulary,"w") as f:
    for key,value in result.items():
        f.write("{}\n".format(key))




