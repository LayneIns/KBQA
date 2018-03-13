import tensorflow as tf 


# Model from Improved Neural Relation Detection 
# for Knowledge Base Question Answering
class HrbilstmModel:

	def __init__(self, arg_config):
		self.relation_seg_input = tf.placeholder(tf.int64, [None, arg_config.max_length_dict['seg_max_length']])
		self.relation_seg_all_input = tf.placeholder(tf.int64, [None, arg_config.max_length_dict['seg_all_max_length']])
		self.question_input = tf.placeholder(tf.float32, [None, arg_config.max_length_dict['question_max_length']])

		self.sequence_length_seg = tf.placeholder(tf.int64, [None,])
		self.sequence_length_seg_all = tf.placeholder(tf.int64, [None])
		self.sequence_length_question = tf.placeholder(tf.int64, [None,])

		self.label_sequence = tf.placeholder(tf.int64, [None,])
		self.question_embedding_matrix = tf.placeholder(tf.float32, [arg_config.question_vocab_size, arg_config.word_embedding_size])

		with tf.variable_scope("relation_embedding_layer"):
			W = tf.get_variable("W", [arg_config.relation_vocab_size, arg_config.relation_embedding_size], initializer=tf.random_normal_initializer(mean=0, stddev=1))
			self.relation_seg_embedding = tf.nn.embedding_lookup(W, self.relation_seg_input)
			self.relation_seg_all_embedding = tf.nn.embedding_lookup(W, self.relation_seg_all_input)

		with tf.variable_scope("BiLSTM_embedding_layer_for_relation", reuse=tf.AUTO_REUSE):
			relation_seg_output_state_fw, relation_seg_output_state_bw = self.BiLstm_for_relation(relation_seg_embedding, sequence_length_seg, arg_config.relation_embedding_size)
			relation_seg_all_output_state_fw, relation_seg_all_output_state_bw = self.BiLstm_for_relation(relation_seg_all_embedding, sequence_length_seg_all, arg_config.relation_embedding_size)

			self.relation_seg_repre = tf.concat([relation_seg_output_state_fw, relation_seg_all_output_state_bw], axis=1)
			self.relation_seg_all_repre = tf.concat([relation_seg_all_output_state_fw, relation_seg_all_output_state_bw], axis=1)

			self.relation_repre = tf.concat([self.relation_seg_repre, self.relation_seg_all_repre], axis=1)

		with tf.variable_scope("word_embedding_layer"):
			W = tf.Variable(tf.constant(0.0,shape=[arg_config.question_vocab_size, arg_config.word_embedding_size]), trainable=False)
			self.question_word_embedding_matrix = tf.assign(W, self.question_embedding_matrix)
			self.question_embedding = tf.nn.embedding_lookup(W, self.question_input)

		with tf.variable_scope("BiLstm_embedding_layer_for_question_1"):
			lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(arg_config.question_embedding_size)
			lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(arg_config.question_embedding_size)

			(output_fw_1, output_bw_1), (output_state_fw_1, output_state_bw_1) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, input=self.question_embedding, sequence_length=self.sequence_length_question, dtype=tf.float32)

			self.question_repre_1 = tf.concat([output_state_fw_1, output_state_bw_1], axis=1)

		with tf.variable_scope("BiLstm_embedding_layer_for_question_2"):
			lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(arg_config.question_embedding_size)
			lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(arg_config.question_embedding_size)


			(output_fw_2, output_bw_2), (output_state_fw_2, output_state_bw_2) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, input=output_fw_1,sequence_length=self.sequence_length_question, dtype=tf.float32)

			self.question_repre_2 = tf.concat([output_state_fw_2, output_state_bw_2], axis=1)

		self.question_repre = tf.add(self.question_repre_1, question_repre_2)

		self.normalize_question_repre = tf.nn.l2_normalize(self.question_repre, 1)        
		self.normalize_relation_repre = tf.nn.l2_normalize(self.relation_repre, 1)

		self.cos_similarity = tf.reduce_sum(tf.multiply(self.normalize_question_repre, self.normalize_relation_repre))
		self.score = tf.reduce_sum(arg_config.gamma + tf.multiply(self.cos_similarity, self.label_sequence), axis=1)
		self.loss = tf.maximum(0, self.score)

		self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss)

		



	def BiLstm_for_relation(self, input, sequence_length, lstm_size):
		lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(lstm_size)
		lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(lstm_size)

		(output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, input=input, sequence_length = sequence_length, dtype=tf.float32)

		return output_state_fw, output_state_bw


		