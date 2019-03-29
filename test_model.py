# -*- coding:UTF-8 -*-
# model_path = "train-check-point-final"
import numpy as np
import tensorflow as tf
import parameter
import data_class
import vocab_class
import datetime
import category_class
import model_class
def test(hbs,model_path,test_seg_file,vocab,category,num_classes,is_training):
	input_obj = data_class.Data(hbs,vocab,test_seg_file,category,hps.num_timesteps,False)
	model  = model_class.Model(hbs,input_obj,vocab.size(),num_classes,is_training)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		thread = tf.train.start_queue_runners(coord=coord)
		saver.restore(sess,model_path)

		#the number of batch size for test
		num_acc_batch = 30
		#when the batch equal the check_batch_idx,outputing the real label and predict label
		check_batch_idx = 25
		acc_check_thresh = 5
		accuracy = []
		for batch in range(num_acc_batch):
			#load data from test seg file
			input_data,output_data  = input_obj.next_batch(hps.batch_size)
			model.update_data(sess,input_data,output_data)
			if batch == check_batch_idx:
				pred,acc = sess.run([model.y_pred,model.accuracy])
				pred_label = category.id2category(pred)
				real_label = category.id2category(output_data)
				print("pred_label vs real_label")
				print(" ".join(pred_label))
				print(" ".join(real_label))
				with open("./data/cnews-predict.txt",'w',encoding='UTF-8') as f:
					for num in range(len(input_data)):
						content = vocab.id2sentence(input_data[num])
						cate_pre = pred_label[num]
						real_la = real_label[num]
						f.write("p=%s\tr=%s\t%s\n" % (cate_pre,real_la,content))
			else:
				pred,acc = sess.run([model.y_pred,model.accuracy])
			if batch > acc_check_thresh:
				accuracy.append(acc)
		print("Average accuracy for test data is {:.3f}".format(np.mean(accuracy)))
		coord.request_stop()
		coord.join(thread)

if __name__ == "__main__":
	test_data = './data/cnews.seg.test.txt'
	vocab_file = "./data/cnews.vocab.txt"
	category_file = "./data/cnews.category.txt"
	load_file = "train-check-point-final"
	model_path = "./model"+"/"+load_file
	hps = parameter.get_paramter_config()
	category = category_class.Category(category_file)
	vocab = vocab_class.Vocab(vocab_file,hps.num_word_threshold)
	num_classes = category.size()
	test(hps,model_path,test_data,vocab,category,num_classes,False)