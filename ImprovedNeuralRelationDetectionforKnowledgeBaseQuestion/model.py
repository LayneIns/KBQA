# -*- coding:utf-8 -*-

import tensorflow as tf 


# Model from Improved Neural Relation Detection 
# for Knowledge Base Question Answering
class HrbilstmModel:

	def __init__(self, arg_config):
		self.relation_seg_input = tf.placeholder(tf.int64, [None, arg_config.max_length_dict['seg_max_length']])
		# [case_number, seg_max_length]
		self.relation_seg_all_input = tf.placeholder(tf.int64, [None, arg_config.max_length_dict['seg_all_max_length']])
		# [case_number, seg_all_max_length]
		self.question_input = tf.placeholder(tf.int64, [None, arg_config.max_length_dict['question_max_length']])
		# [batch_size, question_max_length]

		self.sequence_length_seg = tf.placeholder(tf.int64, [None,])
		# [case_number, ]
		self.sequence_length_seg_all = tf.placeholder(tf.int64, [None])
		# [case_number, ]
		self.sequence_length_question = tf.placeholder(tf.int64, [None,])
		# [batch_size, ]

		self.question_embedding_matrix = tf.placeholder(tf.float32, [arg_config.question_vocab_size, arg_config.word_embedding_size])
		# [question_word_vocab_size, word_embedding_size]

		self.question_matrix = tf.placeholder(tf.float32, [None, None])
		# extend question batch_size to case_number, [case_number, batch_size]
		self.similarity_matrix = tf.placeholder(tf.float32, [None, None])
		# help to calculate the similarity

		with tf.variable_scope("relation_embedding_layer"):
			W = tf.get_variable("W", [arg_config.relation_vocab_size, arg_config.relation_embedding_size], initializer=tf.random_normal_initializer(mean=0, stddev=1))
			self.relation_seg_embedding = tf.nn.embedding_lookup(W, self.relation_seg_input)
			# [one_case_length, seg_max_length, relation_embedding_size]
			self.relation_seg_all_embedding = tf.nn.embedding_lookup(W, self.relation_seg_all_input)
			# [one_case_length, seg_max_length, relation_embedding_size]

		with tf.variable_scope("BiLSTM_embedding_layer_for_relation", reuse=tf.AUTO_REUSE):
			relation_seg_output_state_fw, relation_seg_output_state_bw = self.BiLstm_for_relation(self.relation_seg_embedding, self.sequence_length_seg, arg_config.relation_embedding_size)
			# [one_case_length, relation_embedding_size]
			relation_seg_all_output_state_fw, relation_seg_all_output_state_bw = self.BiLstm_for_relation(self.relation_seg_all_embedding, self.sequence_length_seg_all, arg_config.relation_embedding_size)
			# [one_case_length, relation_embedding_size]

			self.relation_seg_repre = tf.concat([relation_seg_output_state_fw.h, relation_seg_output_state_bw.h], axis=1)
			# [one_case_length, relation_embedding_size*2]
			self.relation_seg_all_repre = tf.concat([relation_seg_all_output_state_fw.h, relation_seg_all_output_state_bw.h], axis=1)
			# [one_case_length, relation_embedding_size*2]

			self.relation_repre = tf.concat([self.relation_seg_repre, self.relation_seg_all_repre], axis=1)
			# [one_case_length, relation_embedding_size*4]

		with tf.variable_scope("word_embedding_layer"):
			W = tf.Variable(tf.constant(0.0, shape=[arg_config.question_vocab_size, arg_config.word_embedding_size]), trainable=False)
			self.question_word_embedding_matrix = tf.assign(W, self.question_embedding_matrix)
			# [question_vocab_size, word_embedding_size]
			self.question_embedding = tf.nn.embedding_lookup(W, self.question_input)
			# [one_case_length, question_max_length, word_embedding_size]

		with tf.variable_scope("BiLstm_embedding_layer_for_question_1"):
			lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(arg_config.question_embedding_size)
			lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(arg_config.question_embedding_size)

			(output_fw_1, output_bw_1), (output_state_fw_1, output_state_bw_1) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs=self.question_embedding, sequence_length=self.sequence_length_question, dtype=tf.float32)
			# output_fw_1 => [one_case_length, question_max_length, question_embedding_size]
			# output_state_fw_1 => [one_case_length, question_embedding_size]

			self.question_repre_1 = tf.concat([output_state_fw_1.h, output_state_bw_1.h], axis=1)
			# [one_case_length, question_embedding_size*2]

		with tf.variable_scope("BiLstm_embedding_layer_for_question_2"):
			lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(arg_config.question_embedding_size)
			lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(arg_config.question_embedding_size)


			(output_fw_2, output_bw_2), (output_state_fw_2, output_state_bw_2) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs=output_fw_1,sequence_length=self.sequence_length_question, dtype=tf.float32)
			# output_state_fw_2 => [one_case_length, question_embedding_size]

			# self.test_element = output_state_fw_2
			self.question_repre_2 = tf.concat([output_state_fw_2.h, output_state_bw_2.h], axis=1)
			# [one_case_length, question_embedding_size*2]

		
		self.question_repre = tf.add(self.question_repre_1, self.question_repre_2)
		self.extended_question_repre = tf.matmul(self.question_matrix, self.question_repre)
		# [one_case_length, question_embedding_size*2]

		self.normalize_question_repre = tf.nn.l2_normalize(self.extended_question_repre, 1)  
		# [one_case_length, question_embedding_size*2]      
		self.normalize_relation_repre = tf.nn.l2_normalize(self.relation_repre, 1)
		# [one_case_length, question_embedding_size*2]

		self.cos_similarity = tf.reduce_sum(tf.multiply(self.normalize_question_repre, self.normalize_relation_repre), axis=1)
		# [1, one_case_length]
		self.cos_similarity_expand = tf.expand_dims(self.cos_similarity, 0)
		
		self.similarity_differ = tf.matmul(self.cos_similarity_expand, self.similarity_matrix)
		self.loss_op = tf.add(arg_config.gamma, self.similarity_differ)
		
		self.loss = tf.reduce_sum(tf.maximum(0.0, self.loss_op))

		self.optimizer = tf.train.AdamOptimizer(learning_rate=arg_config.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss)
		
		self.init = tf.global_variables_initializer()
		



	def BiLstm_for_relation(self, input, sequence_length, lstm_size):
		lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(lstm_size)
		lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(lstm_size)

		(output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs=input, sequence_length = sequence_length, dtype=tf.float32)

		return output_state_fw, output_state_bw


		