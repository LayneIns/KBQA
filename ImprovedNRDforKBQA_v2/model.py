# -*- coding:utf-8 -*-

import tensorflow as tf 

class HrbilstmModel:
	def __init__(self, arg_config):
		self.relation_seg_input = tf.placeholder(tf.int32, [None, arg_config.max_length_dict['seg_max_length']])
		# [case_number, seg_max_length]
		self.relation_seg_all_input = tf.placeholder(tf.int32, [None, arg_config.max_length_dict['seg_all_max_length']])
		# [case_number, seg_all_max_length]
		self.question_input = tf.placeholder(tf.int32, [None, arg_config.max_length_dict['question_max_length']])
		# [case_number, question_max_length]
		
		self.seg_sequence_length = tf.placeholder(tf.int32, [None, ])
		self.seg_all_sequence_length = tf.placeholder(tf.int32, [None, ])
		self.question_sequence_length = tf.placeholder(tf.int32, [None, ])
		
		with tf.variable_scope("relation_embedding_layer"):
			W = tf.get_variable("W", [arg_config.relation_vocab_size, arg_config.relation_embedding_size], initializer=tf.random_normal_initializer(mean=0, stddev=1))
			self.relation_seg_embedding = tf.nn.embedding_lookup(W, self.relation_seg_input)
			# [case_number, seg_max_length, relation_embedding_size]
			self.relation_seg_all_embedding = tf.nn.embedding_lookup(W, self.relation_seg_all_input)
			# [case_number, seg_max_length, relation_embedding_size]
		
		
			
		
		
		
		
		
		
		
		
		
		