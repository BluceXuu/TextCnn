# -*- coding:UTF-8 -*-
import numpy as np
class Data(object):
	"""input data from seg file"""
	def __init__(self, hps,vocab, file,category,num_timesteps,is_shuffle):

		self._vocab = vocab
		self._category = category
		self._num_timesteps = num_timesteps
		self._input = []
		self._output = []
		self._indicator = 0
		self._is_shuffle = is_shuffle
		self._spares_file(file)
	def _spares_file(self,file):
		"""convert seg file to id"""
		with open(file,'r',encoding='utf-8') as f:
			lines = f.readlines()
		for line in lines:
			label,contents = line.strip("\r\n").split("\t")
			id_label = self._category.category2id(label)
			id_sentence = self._vocab.sentence2id(contents)
			id_sentence = id_sentence[0:self._num_timesteps]
			pad_num = self._num_timesteps - len(id_sentence)
			id_sentence = id_sentence + [self._vocab.unk for i in range(pad_num)]
			self._input.append(id_sentence)
			self._output.append(id_label)
		self._input = np.asarray(self._input,dtype=np.int32)
		self._output = np.asarray(self._output,dtype=np.int32)
		if self._is_shuffle:
			self._shuffle_data()

	def size(self):
		#get total number of data
		return len(self._input)

	def _shuffle_data(self):
		"""break up the relationship between train data"""
		p = np.random.permutation(len(self._input))
		self._input = self._input[p]
		self._output = self._output[p]

	def next_batch(self,batch_size):
		"""get batch size data for train/test"""
		endicator = self._indicator + batch_size
		if endicator > len(self._input):
			if self._is_shuffle:
				self._shuffle_data()
			self._indicator = 0
			endicator = batch_size
		if endicator > len(self._input):
			raise Exception("the batch_size is more the all samples")

		batch_data = self._input[self._indicator:endicator]
		batch_label = self._output[self._indicator:endicator]
		self._indicator = endicator
		return batch_data,batch_label