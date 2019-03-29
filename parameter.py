# -*- coding:UTF-8 -*-
import tensorflow as tf

def get_paramter_config():
	"""paramters in cnn"""
	return tf.contrib.training.HParams(
		num_embeding_size = 128,
		num_timesteps = 200,
		num_filters = 64,
		num_kernel_size = 3,
		num_fc_nodes = 100,
		batch_size = 100,
		num_word_threshold = 10,
		num_epochs = 20,
		model_save_names = 'train-check-point'
		)