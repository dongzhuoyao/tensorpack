# Author: Tao Hu <taohu620@gmail.com>

train_path = "/Users/ht/Desktop/annotation_result/data236.txt"
val_path = "/Users/ht/Desktop/annotation_result/ypw.txt"

import os
from collections import Counter

def _build_vocab(caption_list, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(caption_list):
        words = caption.split(' ')  # caption contrains only lower-case words
        for w in words:
            counter[w] += 1

        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] == threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx

def clean_sentence(caption):
    caption = caption.replace('.', ' . ').replace(',', ' , ').replace("'", "").replace('"', '')
    caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
    caption = " ".join(caption.split())  # replace multiple spaces
    caption = caption.lower()
    return caption

def clean_dir(train_path):
    with open(train_path,"r") as f:
        lines = f.readlines()

    id2desc = {}
    lines.sort()
    for line in lines:
        try:
            line = line.strip("\n")
            line_list = line.split("#")
            file_name = line_list[0]
            id = int(file_name.split("_")[2])
            desc = line_list[1]
            id2desc[id] = desc
        except Exception as e:
            pass

    caption_list = []
    for line in lines:
        try:
            line = line.strip("\n")
            line_list = line.split("#")
            file_name = line_list[0]
            desc = clean_sentence(line_list[1])
            #TODO build dic
            print file_name
            caption_list.append(desc)
            print
        except Exception as e:
            pass
    word_to_idx = _build_vocab(caption_list)
    pass


clean_dir(train_path)

