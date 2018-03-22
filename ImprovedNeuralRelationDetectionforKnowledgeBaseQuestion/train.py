# -*- coding:utf-8 -*-

import tensorflow as tf 
from model import HrbilstmModel
import numpy as np 



def train_model(arg_config, training_data_mgr, testing_data_mgr, valid_data_mgr, word_embedding_matrix):
	model = HrbilstmModel(arg_config)

	saver = tf.train.Saver(max_to_keep=1)

	with tf.Session() as sess:
		
		sess.run(model.init)
		for i in range(arg_config.epoch_num):
			training_data_mgr.initialize_batch_cnt()
			for j in range(0, training_data_mgr.total_batch, arg_config.batch_size):

				relation_seg, relation_seg_all, question, \
						sequence_length_seg, sequence_length_seg_all, sequence_length_question, \
						question_matrix, similarity_matrix, case_num_list = training_data_mgr.next_batch(arg_config.batch_size)
				
				_, loss, repre = sess.run([model.train_op, model.loss, model.relation_repre], \
					feed_dict={model.relation_seg_input: relation_seg, model.relation_seg_all_input: relation_seg_all, \
								model.question_input: question, model.sequence_length_seg: sequence_length_seg, \
								model.sequence_length_seg_all: sequence_length_seg_all, model.sequence_length_question: sequence_length_question, \
								model.question_embedding_matrix: word_embedding_matrix, model.question_matrix: question_matrix, \
								model.similarity_matrix: similarity_matrix})
				# print repre
				# raw_input()
				if j % (arg_config.batch_size * 10) == 0:
					print "training --- epoch number:", str(i), ", step:", str(j), ", loss:", str(loss)

				if j % (arg_config.batch_size * 100) == 0:
					total_case = 0
					right_case = 0
					valid_data_mgr.initialize_batch_cnt()
					for k in range(0, valid_data_mgr.total_batch, 100):
						relation_seg, relation_seg_all, question, \
							sequence_length_seg, sequence_length_seg_all, sequence_length_question, \
							question_matrix, similarity_matrix, case_num_list = valid_data_mgr.next_batch(100)
						cos_similarity = sess.run(model.cos_similarity, \
							feed_dict={model.relation_seg_input: relation_seg, model.relation_seg_all_input: relation_seg_all, \
										model.question_input: question, model.sequence_length_seg: sequence_length_seg, \
										model.sequence_length_seg_all: sequence_length_seg_all, model.sequence_length_question: sequence_length_question, \
										model.question_embedding_matrix: word_embedding_matrix, model.question_matrix: question_matrix, \
										model.similarity_matrix: similarity_matrix})
						total_case += len(case_num_list)
						temp_cnt = 0
						print "k:", k
						print cos_similarity
						raw_input()

						for l in range(len(case_num_list)):
							mid_sim = cos_similarity[temp_cnt: temp_cnt + case_num_list[l]]
							if np.max(mid_sim) == mid_sim[0]:
								right_case += 1
							temp_cnt += case_num_list[l]
					print "There are", float(right_case), "right cases in", float(total_case)
					print "The validation accuracy is", float(right_case)/total_case


				

				# x, y, z = sess.run([model.normalize_question_repre, model.normalize_question_repre, model.cos_similarity], \
				# 	feed_dict={model.relation_seg_input: relation_seg, model.relation_seg_all_input: relation_seg_all, \
				# 				model.question_input: question, model.sequence_length_seg: sequence_length_seg, \
				# 				model.sequence_length_seg_all: sequence_length_seg_all, model.sequence_length_question: sequence_length_question, \
				# 				model.question_embedding_matrix: word_embedding_matrix, model.question_matrix: question_matrix, \
				# 				model.similarity_matrix: similarity_matrix})
				# print x
				# print y
				# print z
				# print len(x)
				# print len(y)
				# raw_input()


