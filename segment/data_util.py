#-*-coding:utf-8-*-
"""
author: Leo
date: 2017-4-23
"""
# Chinese Characters: B(Begin),E(End),M(Middle),S(Single)
from __future__ import unicode_literals  # compatible with python3 unicode

import codecs
import sys
from sys import argv

def character_tagging(input_file, output_file):
	input_data = codecs.open(input_file, 'r', 'utf-8')
	output_data = codecs.open(output_file, 'w', 'utf-8')
	for line in input_data.readlines():
		# 移除字符串的头和尾的空格。strip()方法默认是移除空格的
		word_list = line.strip().split()
		for word in word_list:
			words = word.split("/")
			word = words[0]
			if len(word) == 1:
				output_data.write(word + "\tS\n")
			elif len(word) >= 2:
				output_data.write(word[0] + "\tB\n")
				for w in word[1: len(word)-1]:
					output_data.write(w + "\tM\n")
				output_data.write(word[len(word)-1] + "\tE\n")
		output_data.write("\n")
	input_data.close()
	output_data.close()

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print (argv[0])
		sys.exit(-1)
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	character_tagging(input_file, output_file)
