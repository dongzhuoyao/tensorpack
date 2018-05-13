# Author: Tao Hu <taohu620@gmail.com>

import gensim.downloader as api

info = api.info()  # show info about available models/datasets
model = api.load("word2vec-google-news-300")  # download the model and return as object ready for use
print model.most_similar("cat")
print model.wv['computer']


