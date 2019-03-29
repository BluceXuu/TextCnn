# -*- coding: UTF-8 -*-
import os
import datetime
#generate vocab from train_seg_file
def generate_vocab_file(orginal_file,output_file):

	with open(orginal_file,'r',encoding='UTF-8') as f:
		lines = f.readlines()

	words_dicts = {}
	for line in lines:
		label,content = line.strip('\r\n').split("\t")
		for word in content.split():
			words_dicts.setdefault(word,0)
			words_dicts[word] += 1

	#sort in keys asc
	sort_word_dicts = sorted(words_dicts.items(),key=lambda d:d[1],reverse=True)
	with open(output_file,'w',encoding='UTF-8') as f:
		f.write("<UNK>\t1000000\n")
		for item in sort_word_dicts:
			f.write("%s\t%d\n" %(item[0],item[1]))

data_path = "../data"
#train_seg_file
orginal_file = os.path.join(data_path,'cnews.seg.train.txt')
#have generated vocab file
output_file = os.path.join(data_path,'cnews.vocab.txt')

print("begin to generate vocab file from %s" % orginal_file)
generate_vocab_file(orginal_file,output_file)

print("generage vocab file end!")