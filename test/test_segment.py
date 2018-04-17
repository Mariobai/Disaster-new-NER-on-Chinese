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
# text = '北京信息科技大学是一所综合性的大学'
# text = '杨贵国在彭州市隆丰镇巩固村农家庭院为村民表演'
# text = '冯向杰中国画家代表团下榻在贝尔格莱德市中心的哈西诺饭店'
# text = '美国当局列举巴民族权力机构为拯救巴以谈判所必须履行的义务同时告诫巴方的要求应当实际'
# text = '中研国际医药公司先后为全国35家企业的40余个产品进行进入美国的可行性评定'
# text = '撒哈拉以南非洲'
# segList = segmenter.seg(text)
# text_seg = "/".join(segList)
#
# print(text)
# print(text_seg)



