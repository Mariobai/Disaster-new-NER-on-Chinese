#!/usr/bin/python
# -*- coding:utf-8 -*-
# B, M, E, S: Beginning, Middle, End, Single 4 tags

import sys, os
import CRFPP

# linear chain CRF crf_model path, need str input, convert unicode to str
pkg_path = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = str(os.path.join(pkg_path, "segment/data/crf_model2014"))

class Tokenizer(object):

    def __init__(self, model_path=DEFAULT_MODEL):
        self.model = CRFPP.Tagger("-m " + model_path)
    
    def seg(self, text):
        '''
        text: String, text to be segmented;
        crf_model: path of pretrained CRFPP crf_model,
        '''
        segList = []
        model = self.model
        model.clear()
        # char会每一只拿一个字符第一次拿到"我爱吃北京烤鸭"中的"我"
        for char in text.strip():  # cha`r in String
            char = char.strip()
            if char:
                model.add((char + "\to\tB"))
        model.parse()
        size = model.size()     # 这个会获得模型中所有字符size。上的是8
        xsize = model.xsize()
        word = ""
        for i in range(0, size):
            for j in range(0, xsize):
                char = model.x(i, j)
                tag = model.y2(i)
                if tag == 'B':
                    word = char
                elif tag == 'M':
                    word += char
                elif tag == 'E':
                    word += char
                    segList.append(word)
                    word = ""
                else: # tag == 'S'
                    word = char
                    segList.append(word)
                    word = ""
        return segList

# Create Instance of a tokenizer
# print (DEFAULT_MODEL)
tk = Tokenizer(DEFAULT_MODEL)

# Global functions for call
seg = tk.seg
