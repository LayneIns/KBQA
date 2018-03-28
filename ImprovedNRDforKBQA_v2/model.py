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
		self.cal_matrix = tf.placeholder(tf.float32, [None, None])
		
		with tf.variable_scope("relation_embedding_layer"):
			W = tf.get_variable("W", [arg_config.relation_vocab_size, arg_config.relation_embedding_size], initializer=tf.random_normal_initializer(mean=0, stddev=1))
			self.relation_seg_embedding = tf.nn.embedding_lookup(W, self.relation_seg_input)
			# [case_number, seg_max_length, relation_embedding_size]
			self.relation_seg_all_embedding = tf.nn.embedding_lookup(W, self.relation_seg_all_input)
			# [case_number, seg_all_max_length, relation_embedding_size]
		
		with tf.variable_scope("relation_BiLSTM_Layer") as scope:
			self.relation_seg_output_fw, self.relation_seg_output_bw = self.relationBiLSTM(self.relation_seg_embedding, arg_config.relation_lstm_size, self.seg_sequence_length)
			scope.reuse_variables()
			self.relation_seg_all_output_fw, self.relation_seg_all_output_bw = self.relationBiLSTM(self.relation_seg_all_embedding, arg_config.relation_lstm_size, self.seg_all_sequence_length)
		
		self.relation_seg_embedding = tf.concat([self.relation_seg_output_fw, self.relation_seg_output_bw], axis=2)
		# [case_number, seg_max_length, relation_lstm_size*2]
		self.relation_seg_all_embedding = tf.concat([self.relation_seg_all_output_fw, self.relation_seg_all_output_bw], axis=2)
		# [case_number, seg_all_max_length, relation_lstm_size*2]
		
		self.relation_lstm_embedding = tf.concat([self.relation_seg_embedding, self.relation_seg_all_embedding], axis=1)
		# [case_number, seg_max_length+seg_all_max_length, relation_lstm_size*2]
		self.relation_lstm_embedding_expand = tf.expand_dims(self.relation_lstm_embedding, -1)
		# [case_number, seg_max_length+seg_all_max_length, relation_lstm_size*2, 1]
		
		self.relation_embedding = tf.nn.max_pool(
			self.relation_lstm_embedding_expand, 
			ksize = arg_config.relation_ksize, 
			strides = [1, 1, 1, 1], 
			padding = "VALID"
		)
		# [case_number, 1, relation_lstm_size*2, 1]
		self.relation_embedding_squeeze = tf.squeeze(self.relation_embedding)
		# [case_number, relation_lstm_size*2]
		
		with tf.variable_scope("question_embedding_layer"):
			W = tf.get_variable("W", [arg_config.question_vocab_size, arg_config.question_embedding_size], initializer=tf.random_normal_initializer(mean=0, stddev=1))
			self.question_word_embedding = tf.nn.embedding_lookup(W, self.question_input)
			# [case_number, question_max_length, question_embedding_size]
		
		with tf.variable_scope("question_BiLSTM_layer"):
			lstm_cell_fw = []
			lstm_cell_bw = []
			for i in range(2):
				lstm_cell_fw.append(self.get_lstm_cell(arg_config.question_lstm_size))
				lstm_cell_bw.append(self.get_lstm_cell(arg_config.question_lstm_size))
			self.question_outputs, self.question_output_state_fw, self.question_output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs=self.question_word_embedding, sequence_length=self.question_sequence_length, dtype=tf.float32)
		
		self.question_lstm_embedding_expand = tf.expand_dims(self.question_outputs, -1)
		# [case_number, question_max_length, question_lstm_size*2, 1]
		self.question_embedding = tf.nn.max_pool(
			self.question_lstm_embedding_expand, 
			ksize = arg_config.question_ksize, 
			strides = [1, 1, 1, 1], 
			padding = "VALID"
		)
		# [case_number, 1, question_lstm_size*2, 1]
		self.question_embedding_squeeze = tf.squeeze(self.question_embedding)
		# [case_number, question_lstm_size*2]
		
		self.cosine_similarity = self.calculate_cosine_similarity(self.relation_embedding_squeeze, self.question_embedding_squeeze)
		self.cosine_similarity_expand = tf.expand_dims(self.cosine_similarity, -1)
		self.sub_res = tf.matmul(self.cal_matrix, self.cosine_similarity_expand)
		self.sub_res_squeeze = tf.squeeze(self.sub_res)
		
		self.hinge_loss = tf.maximum(0.0, tf.add(arg_config.gamma, self.sub_res_squeeze))
		self.loss = tf.reduce_sum(self.hinge_loss)
		
		self.optimizer = tf.train.AdamOptimizer(learning_rate=arg_config.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss)		
		
		self.init = tf.global_variables_initializer()

	
	def calculate_cosine_similarity(self, relation, question):
		len_pool_q = tf.sqrt(tf.reduce_sum(tf.pow(question, 2), [1]))
		len_pool_r = tf.sqrt(tf.reduce_sum(tf.pow(relation, 2), [1]))
		
		q_r_cosine = tf.div(tf.reduce_sum(tf.multiply(relation, question), [1]), tf.multiply(len_pool_q, len_pool_r))
		
		return q_r_cosine
		

		
			
	def get_lstm_cell(self, hidden_unit):
		single_cell = tf.contrib.rnn.BasicLSTMCell(hidden_unit)
		return single_cell
		
	def relationBiLSTM(self, input, lstm_size, sequence_length):
		lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(lstm_size)
		lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(lstm_size)
		
		(output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs=input, sequence_length = sequence_length, dtype=tf.float32)
		
		return output_fw, output_bw
		
			
		
		
		
		
		
		
		
		
		
		