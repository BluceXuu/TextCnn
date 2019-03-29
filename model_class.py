import numpy as np
import tensorflow as tf
import math
class Model(object):
	"""contrust text-cnn model"""
	def __init__(self,hps,inputs_obj,vocab_size,num_classes,is_training,dropout=0.5,init_scale=0.05):
		self.embeding_size = hps.num_embeding_size
		self.inputs_obj = inputs_obj
		self.vocab_size = vocab_size
		self.num_classes = num_classes
		self.lw_data = tf.Variable(tf.zeros([hps.batch_size,hps.num_timesteps],dtype=tf.int32),trainable=False)
		self.lw_label = tf.Variable(tf.zeros([hps.batch_size],dtype=tf.int32),trainable=False)
		self.input_data = tf.placeholder(tf.int32,[hps.batch_size,hps.num_timesteps])
		self.input_label = tf.placeholder(tf.int32,[hps.batch_size])
		self.lr_d = tf.assign(self.lw_data,self.input_data)
		self.lr_l = tf.assign(self.lw_label,self.input_label)
		#embed the input data
		embeding_initial = tf.random_uniform_initializer(-init_scale,init_scale)
		with tf.variable_scope("embeding",initializer=embeding_initial):
			embedings = tf.get_variable("embeding",[self.vocab_size,self.embeding_size],tf.float32)
			inputs = tf.nn.embedding_lookup(embedings,self.lw_data)
		#define drop out layer
		if is_training and dropout < 1:
			inputs = tf.nn.dropout(inputs,dropout)
		init_scale = 1.0/math.sqrt(hps.num_embeding_size+hps.num_filters)/3.0
		cnn_initial = tf.random_uniform_initializer(-init_scale,init_scale)

		#define cnn layer
		with tf.variable_scope("cnn",initializer=cnn_initial):
			cond1 = tf.layers.conv1d(inputs,hps.num_filters,hps.num_kernel_size,activation=tf.nn.relu)
			global_maxpooling = tf.reduce_max(cond1,axis=[1])

		#define fc
		init_fc = tf.uniform_unit_scaling_initializer(factor=1.0)
		with tf.variable_scope("fc",initializer=init_fc):
			fc1 = tf.layers.dense(global_maxpooling,hps.num_fc_nodes,name='fc1',activation=tf.nn.relu)
			if is_training and dropout < 1:
				fc1 = tf.contrib.layers.dropout(fc1,dropout)
			logits = tf.layers.dense(fc1,self.num_classes)

		with tf.name_scope("metrics"):
			softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.lw_label)
			self.loss = tf.reduce_mean(softmax_loss)
			self.y_pred = tf.argmax(tf.nn.softmax(logits),axis=1)
			correct_y = tf.equal(tf.cast(self.y_pred,tf.int32),self.lw_label)
			self.accuracy = tf.reduce_mean(tf.cast(correct_y,tf.float32))

		if not is_training:
			return
		#clip the gradient in order to gradient explode
		self.learn_rate = tf.Variable(0.0,trainable=False)
		tvars = tf.trainable_variables()
		grad,_ = tf.clip_by_global_norm(tf.gradients(self.loss,tvars),5)
		optimizer = tf.train.GradientDescentOptimizer(self.learn_rate)
		self.train_op = optimizer.apply_gradients(zip(grad,tvars),global_step=tf.train.get_or_create_global_step())

		#update learning rate
		self.new_lr = tf.placeholder(tf.float32,[])
		self.lr_update = tf.assign(self.learn_rate,self.new_lr)

	def update_lr(self,sess,new_lr):
		sess.run(self.lr_update,feed_dict={self.new_lr:new_lr})

	def update_data(self,sess,input_data,output_data):
		sess.run([self.lr_d,self.lr_l],feed_dict={self.input_data:input_data,self.input_label:output_data})





