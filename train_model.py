# -*- coding:UTF-8 -*-
import numpy as np
import tensorflow as tf
import parameter
import data_class
import vocab_class
import datetime
import category_class
import model_class

"""
hbs:parameter in train/test
train_data:train data file
vocab:voca object
num_classes:total data labels
is_training:if the value is True,it wii train;if the value is false,it wii test
num_epochs:total epch
model_save_names:model parameter file name
learning_rate=1.0 
max_lr_epoch=10 the learning rate wii change each max_lr_epoch when training
lr_decay=0.93 learning rate decay rate
print_iter=100 the console wii output the result each 100 iterations
"""
def train(hbs,train_data,vocab,category,num_classes,is_training,num_epochs,model_save_names,learning_rate=1.0,max_lr_epoch=10,lr_decay=0.93,print_iter=100):
	input_obj = data_class.Data(hbs,vocab,train_data,category,hps.num_timesteps,True)
	save_path = "./save"
	model  = model_class.Model(hbs,input_obj,vocab.size(),num_classes,is_training)
	init = tf.global_variables_initializer()
	origi_decay = lr_decay
	with tf.Session() as sess:
		sess.run(init)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		saver = tf.train.Saver(max_to_keep=5)
		#start to train
		for epoch in range(num_epochs):
			new_lr_decay = origi_decay**(max(epoch+1-max_lr_epoch,0.0))
			model.update_lr(sess,new_lr_decay*learning_rate)

			current_time = datetime.datetime.now()
			epoch_size = input_obj.size() // hbs.batch_size
			for step in range(epoch_size):
				#load data from train seg file
				input_data,output_data  = input_obj.next_batch(hps.batch_size)
				model.update_data(sess,input_data,output_data)
				if step % print_iter != 0:
					cost,_ = sess.run([model.loss,model.train_op])
				else:
					seconds = (float((datetime.datetime.now()-current_time).seconds)) / print_iter
					current_time = datetime.datetime.now()
					cost,_,acc = sess.run([model.loss,model.train_op,model.accuracy])
					print("Epoch{}, Step{}, Cost{:.3f}, Accuracy{:.3f}, Seconds per step:{:.3f}".format(epoch,
                                                                                                        step,cost,acc,seconds))
			saver.save(sess,save_path + '/'+model_save_names,global_step=epoch)
		saver.save(sess,save_path + '/'+model_save_names+ '-final')
		coord.request_stop()
		coord.join(threads)

if __name__ == "__main__":
	train_data = './data/cnews.seg.train.txt'
	vocab_file = "./data/cnews.vocab.txt"
	category_file = "./data/cnews.category.txt"
	hps = parameter.get_paramter_config()
	category = category_class.Category(category_file)
	vocab = vocab_class.Vocab(vocab_file,hps.num_word_threshold)
	num_classes = category.size()
	train(hps,train_data,vocab,category,num_classes,True,hps.num_epochs,hps.model_save_names)