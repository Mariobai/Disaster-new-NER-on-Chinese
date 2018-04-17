# encoding: utf-8
from __future__ import unicode_literals

import codecs
import jieba
from news_seg import segmenter


# input_path = codecs.open('../seg_data/train_disaster.txt', 'r', encoding='utf-8')
# out_path = codecs.open('../seg_data/seg_disaster.txt', 'w', encoding='utf-8')

def seg_data(filename):
    """分词方法"""
    words = []
    with filename as ip:
        for text in ip:
            seg = segmenter.seg(text)
            words.append(seg)
    return words

# text = u"2018年6月7日5分，四川地震造成了10亿元的经济损失"
# segList = segmenter.seg(text)
# text_seg = "/".join(segList)
#
# print(text)
# print(text_seg)



