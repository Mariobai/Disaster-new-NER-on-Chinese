# encoding: utf-8
import sys
import codecs
"""
  通过与黄金标准文件对比分析中文分词效果.

  使用方法：
          python crf_tag_score.py test_gold.utf8 your_tagger_output.utf8

  分析结果示例如下:
    标准词数：104372 个，正确词数：96211 个，错误词数：6037 个
    标准行数：1944，正确行数：589 ，错误行数：1355
    Recall: 92.1808531024%
    Precision: 94.0957280338%
    F MEASURE: 93.1284483593%


  参考：中文分词器分词效果的评测方法
  http://ju.outofmemory.cn/entry/46140

"""


def read_line(f):
    '''
        读取一行，并清洗空格和换行
    '''
    line = f.readline()
    line = line.strip('\n').strip('\r').strip(' ')
    while (line.find('  ') >= 0):
        line = line.replace('  ', ' ')
    return line

if __name__ == "__main__":
    if len(sys.argv)!=3:
        print '  用法：crf_score.py test_gold.utf8 your_tagger_output.utf8'.decode('utf8')
        sys.exit(1)

    file_gold = codecs.open(sys.argv[1], 'r', 'utf8')
    file_tag = codecs.open(sys.argv[2], 'r', 'utf8')

    line1 = read_line(file_gold)
    N_count = 0
    e_count = 0
    c_count = 0
    e_line_count = 0
    c_line_count = 0

    while line1:
        line2 = read_line(file_tag)

        list1 = line1.split(' ')
        list2 = line2.split(' ')

        count1 = len(list1)   # 标准分词数
        N_count += count1
        if line1 == line2:
            c_line_count += 1
            c_count += count1
        else:
            e_line_count += 1
            count2 = len(list2)

            arr1 = []
            arr2 = []

            pos = 0
            for w in list1:
                arr1.append(tuple([pos, pos + len(w)]))
                pos += len(w)

            pos = 0
            for w in list2:
                arr2.append(tuple([pos, pos + len(w)]))
                pos += len(w)

            for tp in arr2:
                if tp in arr1:
                    c_count += 1
                else:
                    e_count += 1

        line1 = read_line(file_gold)

    R = c_count * 100. / N_count
    P = c_count * 100. / (c_count + e_count)
    F = 2. * P * R / (P + R)
    ER = 1. * e_count / N_count

    print '  标准词数：{} 个，正确词数：{} 个，错误词数：{} 个'.format(N_count, c_count, e_count).decode('utf8')
    print '  标准行数：{}，正确行数：{} ，错误行数：{}'.format(c_line_count+e_line_count, c_line_count, e_line_count).decode('utf8')
    print '  Recall: {}%'.format(R)
    print '  Precision: {}%'.format(P)
    print '  F MEASURE: {}%'.format(F)
    print '  ERR RATE: {}%'.format(ER)