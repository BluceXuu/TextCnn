#-*- coding: UTF-8 -*-
import jieba
import os
data_path = "../data"

#产生分词文件
def generate_seg_file(origin_file,seg_file):

	with open(origin_file,'r',encoding='utf-8') as f:
		lines = f.readlines()

	with open(seg_file,'w',encoding='utf-8') as f:
		for line in lines:
			label,content = line.strip('\r\n').split("\t")
			word_iters = jieba.cut(content)
			word_contents = ""
			for word in word_iters:
				word = word.strip(" ")
				if word != "":
					word_contents += word + ' '
			output_line = "%s\t%s\n" %(label,word_contents.strip(' '))
			f.write(output_line)

# 原始数据文件
train_data_file = os.path.join(data_path,'cnews.train.txt')
val_data_file = os.path.join(data_path,'cnews.val.txt')
test_data_file = os.path.join(data_path,'cnews.test.txt')

#分词后的文件
seg_train_file = os.path.join(data_path,'cnews.seg.train.txt')
seg_val_file = os.path.join(data_path,'cnews.seg.val.txt')
seg_test_file = os.path.join(data_path,'cnews.seg.test.txt')

#生成分词文件
generate_seg_file(train_data_file,seg_train_file)
generate_seg_file(val_data_file,seg_val_file)
generate_seg_file(test_data_file,seg_test_file)