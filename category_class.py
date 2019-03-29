# -*- coding:UTF-8 -*-
import numpy as np
class Category(object):
	"""convert category to id"""
	def __init__(self,category_file):
		self._category_to_id = {}
		with open(category_file,'r',encoding='UTF-8') as f:
			lines = f.readlines()
		for line in lines:
			label = line.strip("\r\n")
			idx = len(self._category_to_id)
			self._category_to_id[label] = idx

	def size(self):
		return len(self._category_to_id)
	
	def category2id(self,category):
		"""convert category to id"""
		if not category in self._category_to_id:
			raise Exception("%s is not in category" % category)
		return self._category_to_id[category]
	def id2category(self,id_list):
		idx_to_lable = dict(zip(self._category_to_id.values(),self._category_to_id.keys()))
		return [idx_to_lable[idx] for idx in id_list]
category_file = "./data/cnews.category.txt"
category = Category(category_file)
print(category.size())