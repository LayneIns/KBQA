# -*- coding:utf-8 -*-

import tensorflow as tf 
from model import HrbilstmModel
import numpy as np 

def train_model(arg_config, training_data_mgr, testing_data_mgr, valid_data_mgr):
	model = HrbilstmModel(arg_config)
	saver = tf.train.Saver(max_to_keep=1)
	
	with tf.Session() as sess:

		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(model.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		sess.run(tf.global_variables_initializer())
		
		sess.run(model.init)
		for i in range(arg_config.epoch_num):
			training_data_mgr.initialize_batch_cnt()
			batch_number = 0
			
			while True:
				relation_seg, relation_seg_all, question, seg_sequence_list, seg_all_sequence_list, question_sequence_list, ret_case_num, cal_matrix = training_data_mgr.next_batch(arg_config.batch_size)

				if len(relation_seg) == 0:
					break
				
				
				_, loss = sess.run([train_op, model.loss], feed_dict={model.relation_seg_input: relation_seg, model.relation_seg_all_input: relation_seg_all, model.question_input: question, model.seg_sequence_length: seg_sequence_list, model.seg_all_sequence_length: seg_all_sequence_list, model.question_sequence_length: question_sequence_list, model.cal_matrix: cal_matrix})
				
				if batch_number % 10 == 0:
					cosine_similarity, sub_res = sess.run([model.cosine_similarity, model.sub_res_squeeze], feed_dict={model.relation_seg_input: relation_seg, model.relation_seg_all_input: relation_seg_all, model.question_input: question, model.seg_sequence_length: seg_sequence_list, model.seg_all_sequence_length: seg_all_sequence_list, model.question_sequence_length: question_sequence_list, model.cal_matrix: cal_matrix})
					total_case_number = len(sub_res)
					right_case_number = 0
					for j in range(len(sub_res)):
						if sub_res[j] < 0:
							right_case_number += 1
					
					print "training --- epoch number:", str(i), ", batch number:", str(batch_number), ", loss:", str(loss)
					print "total case:", str(total_case_number), ", right case:", str(right_case_number), ", ratio:", float(right_case_number)/total_case_number
				
				if batch_number % 100 == 0:
					valid_data_mgr.initialize_batch_cnt()
					
					total_case_number = 0
					right_case_number = 0
					total_batch_number = 0
					right_batch_number = 0
					
					while True:
						relation_seg, relation_seg_all, question, seg_sequence_list, seg_all_sequence_list, question_sequence_list, ret_case_num, cal_matrix = valid_data_mgr.next_batch(200)
						
						if len(relation_seg) == 0:
							break
						
						cosine_similarity, sub_res = sess.run([model.cosine_similarity, model.sub_res_squeeze], feed_dict={model.relation_seg_input: relation_seg, model.relation_seg_all_input: relation_seg_all, model.question_input: question, model.seg_sequence_length: seg_sequence_list, model.seg_all_sequence_length: seg_all_sequence_list, model.question_sequence_length: question_sequence_list, model.cal_matrix: cal_matrix})
						
						total_case_number += len(sub_res)
						for j in range(len(sub_res)):
							if sub_res[j] < 0:
								right_case_number += 1
						
						total_batch_number += len(ret_case_num)
						start_point = [0]
						for j in range(len(ret_case_num) - 1):
							start_point.append(start_point[j] + ret_case_num[j])
						
						for j in range(len(ret_case_num)):
							flag = True
							for k in range(1, ret_case_num[j]):
								if cosine_similarity[start_point[j]] < cosine_similarity[start_point[j] + k]:
									flag = False
									break
							if flag:
								right_batch_number += 1
					print "Validation:"
					print "Total case number:", total_case_number, ", right case number:", right_case_number, ", ratio:", float(right_case_number)/total_case_number
					print "Total batch number:", total_batch_number, ", right batch_number:", right_batch_number, ", ratio:", float(right_batch_number)/total_batch_number
				
				
				batch_number += 1
				
	
				

