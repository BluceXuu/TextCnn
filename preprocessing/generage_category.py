# -*- coding: UTF-8 -*-
import os

def generate_category_file(orginal_file,output_file):

	with open(orginal_file,'r',encoding='UTF-8') as f:
		lines = f.readlines()

	category_dict = {}
	for line in lines:
		label,content = line.strip("\r\n").split("\t")
		category_dict.setdefault(label,0)
		category_dict[label] += 1

	with open(output_file,'w',encoding='UTF-8') as f:
		for item in category_dict:
			f.write("%s\n" % item)

data_path = "../data"
orginal_file = os.path.join(data_path,'cnews.seg.train.txt')
output_file = os.path.join(data_path,'cnews.category.txt')
generate_category_file(orginal_file,output_file)