# encoding: utf-8

"""
author: mario
data: 2018/4/2
"""

import codecs
import yaml
import numpy as np

from news_seg.data_process import *
from news_seg.test.test_segment import *
from sklearn.cross_validation import StratifiedKFold, train_test_split

from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import *
from keras.optimizers import Adam
from keras.models import model_from_yaml, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Bidirectional, Reshape, concatenate, Input, merge, Dropout, Flatten
from keras_contrib.layers import CRF


# 定义一些参数
maxlen = 10
embed_dim = 128
n_epoch = 10
batch_sizes = 128
SINGLE_ATTENTION_VECTOR = False

def read_corpuss():
    """读取语料,这里只需要读取word/tag形式的语料就可以了"""
    train_data = codecs.open('./data/combine_news_dis_update.txt', 'r', encoding='utf-8')
    data = []
    with train_data as td:
        for lines in td:
            if lines != '\n':
                split_line = lines.strip().split()
                data.extend(split_line)
    train_data.close()
    return data

def split_word_tags(data):
    """将word/tag形式的语料切分开，分别存储到word和tag列表中"""
    word, tag = [], []
    for word_tag_pair in data:
        # print('word_tag_pair', word_tag_pair)
        pairs = word_tag_pair.split('/')
        try:
            if(len(pairs[0].strip()) != 0 and len(pairs[1].strip()) != 0):
                word.append(pairs[0].strip())
                tag.append(pairs[1].strip())
        except:
            pass
    return word, tag


def train_val_test_split(word, label):
    """划分训练集、验证集、测试集"""
    # 划分训练集和验证集
    x_temp_train, x_val, y_temp_train, y_val = train_test_split(word, label, test_size=0.1, random_state=2018)
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_temp_train, y_temp_train, test_size=0.1, random_state=2018)

    return x_train, y_train, x_val, y_val, x_test, y_test



def bi_gru_crf(word_count, index_embed_matrix):
    """模型定义"""
    inputs = Input(shape=(maxlen, ), name='emb')
    emb = Embedding(input_dim=word_count, output_dim=embed_dim, weights=[index_embed_matrix])(inputs)

    bi_gru_layer = Bidirectional(GRU(128, return_sequences=True, dropout=0.1))(emb)

    bi_gru_flatten = Flatten()(bi_gru_layer)

    bi_gru_drop_layer = Dropout(0.5)(bi_gru_flatten)

    # dense_layer = Dense(30, activation='relu')(bi_gru_drop_layer)

    # crf_layer = CRF(10, sparse_target=True)(dense_layer)
    output = Dense(4, activation='softmax')(bi_gru_drop_layer)

    model = Model(input=inputs, output=output)

    model.summary()

    # optmr = Adam(lr=0.001, beta_1=0.5)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def train():
    """训练方法"""
    word_tag = read_corpuss()
    words, labels = split_word_tags(word_tag)
    # print("labels", labels)
    w2index, w2vec = word2vec_obtain()

    new_words = text2index(w2index, words)


    x_train, x_train_label, x_val, x_val_label, x_test, x_test_label = train_val_test_split(new_words, labels)
    word_count, index_embed_martix = get_data(w2index, w2vec)

    x_train_label = np.array(generate_label(x_train_label))
    # print('x_train_label:', x_train_label)
    x_val_label = np.array(generate_label(x_val_label))
    x_test_label = np.array(generate_label(x_test_label))
    model = bi_gru_crf(word_count, index_embed_martix)
    print('train....')
    checkpoint = ModelCheckpoint('./model_result/ner_dis_update.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    callbacks_list = [checkpoint, early]
    print("fiting...")
    # yaml_string = model.to_yaml()
    # with open('./model_result/ner_dis_update.yml', 'w') as outfile:
    #     outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    # model.fit(x_train, x_train_label, batch_size=batch_sizes, epochs=n_epoch, verbose=1, validation_data=(x_val, x_val_label), callbacks=callbacks_list)
    #
    model.load_weights('./model_result/ner_dis_update.hdf5')
    print('Evaluate...')
    score = model.evaluate(x_test, x_test_label, verbose=1, batch_size=batch_sizes)

    print('Test Score:', score)

def test():
    """这里是用来做预测的"""
    print('loading model....')
    test_path = codecs.open('./seg_data/seg_news.txt', 'r', encoding='utf8')
    with open('./model_result/ner_dis_update.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)
    print('loading weights......')
    model.load_weights('./model_result/ner_dis_update.hdf5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy"])
    test_data = seg_data(test_path)
    print('test_data', test_data)
    w2index, w2vec = word2vec_obtain()
    new_words = []
    for testd in test_data:
        new_word = text2index(w2index, testd)
        new_words.append(new_word)
    print('new_wods', new_words)
    pre_result = []
    for word in new_words:
        result = model.predict(word)
        results = []
        for each in result:
            results.append(np.argmax(each))
        pre_result.append(results)

    loc, time, dis = '', '', ''
    for i in range(len(test_data)):
        for word, pre_tag in zip(test_data[i], pre_result[i]):
            if pre_tag == 0:
                loc += '  ' + word
            elif pre_tag == 1:
                time += '  ' + word
            elif pre_tag == 2:
                dis += '  ' + word

    print(['LOC:' + loc, 'TIME:' + time, 'DIS:' + dis])


if __name__ == '__main__':
    # train()
    test()


