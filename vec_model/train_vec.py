# encoding: utf-8

"""
author: mario
data: 2018/4/2
"""

import jieba
import re
import numpy as np
import multiprocessing
import codecs

import logging


from gensim.models import KeyedVectors
from gensim.models import word2vec


# 定义各种参数
embed_dim = 128
window_size = 7
min_counts = 3
n_iter = 10
cpu_count = multiprocessing.cpu_count()

def read_file():
    """读文件"""
    train_data = codecs.open('../data/combine_seg_data.txt', 'r', encoding='utf-8')
    train_datas = []

    with train_data as td:
        for data in td:
            data = data.strip().split()
            train_datas.extend(data)

    train_data.close()
    return train_datas


# def clean_str(data):
#     """去除句子中的字母"""
#     rm_word = re.sub(r'[a-zA-Z]', '', data)
#     return rm_word

# def seg_word(data):
#     """将训练集中的每个词进行分词处理"""
#     # input_path = codecs.open('../vec_model/seg_data.txt', 'w', encoding='utf-8')
#     seg_words = []
#     for word in data:
#         seg = list(jieba.lcut(word.strip()))
#         # seg_data = ' '.join(seg)
#         seg_words.append(seg)
#         # input_path.write(seg_data+'\n')
#     return seg_words


def word2vec_train():
    """训练词向量"""
    sentences = word2vec.Text8Corpus('../data/combine_seg_data.txt')
    model = word2vec.Word2Vec(sentences, size=embed_dim, sg=1, window=window_size, hs=1, workers=cpu_count, min_count=min_counts)
    model.wv.save_word2vec_format('../vec_model/word2vec_models.txt', binary=0)
    # model = KeyedVectors.load_word2vec_format('../vec_model/word2vec_models.txt', binary=False)
    # print('model[word]', model['我'])
def train():
    """训练模型的方法"""
    # train_data = read_file()
    print('Tokenising...')
    print('Training a Word2vec model...')
    # 拿所有的数据训练词向量
    word2vec_train()

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    train()


























