# -*- coding: utf-8 -*-

import sys
# 用来进行准确率和召回率以及F值得计算的
if __name__ == "__main__":
    try:
        file = open('./data/pku_test.utf8', "r")
    except:
        print("result file is not specified, or open failed!")
        sys.exit()

    wc_of_test = 0
    wc_of_gold = 0
    wc_of_correct = 0
    flag = True

    for l in file:
        if l == '\n': continue

        _, _, g, r = l.strip().split()

        if r != g:
            flag = False

        if r in ('E', 'S'):
            wc_of_test += 1
            if flag:
                wc_of_correct += 1
            flag = True

        if g in ('E', 'S'):
            wc_of_gold += 1

    print("WordCount from test result:", wc_of_test)
    print("WordCount from golden data:", wc_of_gold)
    print("WordCount of correct segs :", wc_of_correct)

    # 查全率
    P = wc_of_correct / float(wc_of_test)
    # 查准率，召回率
    R = wc_of_correct / float(wc_of_gold)

    print("P = %f, R = %f, F-score = %f" % (P, R, (2 * P * R) / (P + R)))