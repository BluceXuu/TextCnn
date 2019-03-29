# -*- coding:UTF-8 -*-
import os

class Vocab(object):
	"""generate vocab and convert sentence to id"""
	
	def __init__(self,vocab_file,num_word_threshold):
		self._word_to_id = {}
		self._unk = -1
		self._num_word_threshold = num_word_threshold
		self._word2id(vocab_file)

	def _word2id(self,vocab_file):
		"""convert word to number from vocab_file"""
		with open(vocab_file,'r',encoding='UTF-8') as f:
			lines = f.readlines()

		for line in lines:
			word,frequency = line.strip("\r\n").split("\t")
			frequency = int(frequency)
			if frequency < self._num_word_threshold:
				continue
			idx = len(self._word_to_id)
			if word == "<UNK>":
				self._unk = idx
			self._word_to_id[word] =idx

	def size(self):
		"""get vocab size"""
		return len(self._word_to_id)

	@property
	def unk(self):
		return self._unk

	def _unword_in_vocab(self,word):
		"""convert the word whitch is not in vocab wo id"""
		return self._word_to_id.get(word,self._unk)

	def sentence2id(self,sentence):
		"""convert train/text/val sentence to id"""
		return [self._unword_in_vocab(word) for word in sentence.split()]

	def id2sentence(self,num_list):
		"""convert id to word"""
		id_to_word = dict(zip(self._word_to_id.values(),self._word_to_id.keys()))
		return "".join(id_to_word[idx] for idx in num_list)

idlist = [0, 1845, 1559, 53, 12823, 6194, 2507, 20320, 4800, 3583, 3, 148]
vocab_file = "./data/cnews.vocab.txt"
vocab = Vocab(vocab_file,40)
print(vocab.id2sentence(idlist))