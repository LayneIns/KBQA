# -*- coding:utf-8 -*-

import numpy as np 


class argConfig():
	def __init__(self, max_length_dict, relation_vocab_size, question_vocab_size):
		self.relation_embedding_size = 150
		self.max_length_dict = max_length_dict
		self.relation_vocab_size = relation_vocab_size
		self.word_embedding_size = 300
		self.question_vocab_size = question_vocab_size
		self.question_embedding_size = 300


class dataMgr():

	def __init__(self, data, max_length_dict):
		self.max_length_dict = max_length_dict
		self.question_list, self.relation_seg_list, self.relation_seg_all_list, \
			self.question_sequence_length_list, self.relation_seg_sequence_length_list, self.relation_seg_all_sequence_length_list, \
			self.case_num_list = self.data_process(data, max_length_dict)
		self.total_batch = len(self.question_list)
		self.batch_cnt = 0
	
	def initialize_batch_cnt(self):
		self.batch_cnt = 0

	def next_batch(self):
		if self.batch_cnt <= self.total_batch - 10:
			question = self.question_list[self.batch_cnt: self.batch_cnt+10]
			
			relation_seg = self.relation_seg_list[self.batch_cnt: self.batch_cnt+10]
			relation_seg_all = self.relation_seg_all_list[self.batch_cnt: self.batch_cnt+10]
			sequence_length_seg = self.relation_seg_sequence_length_list[self.batch_cnt: self.batch_cnt+10]
			sequence_length_seg_all = self.relation_seg_all_sequence_length_list[self.batch_cnt: self.batch_cnt+10]
			sequence_length_question = self.question_sequence_length_list[self.batch_cnt: self.batch_cnt+10]

			question_matrix = self.build_calculate_matrix_for_question(self.case_num_list[self.batch_cnt: self.batch_cnt+10])
			similarity_matrix = self.build_calculate_matrix_for_similarity(self.case_num_list[self.batch_cnt: self.batch_cnt+10])	

			self.batch_cnt += 10
		
		else:
			question = self.question_list[self.batch_cnt: ]
			
			relation_seg = self.relation_seg_list[self.batch_cnt: ]
			relation_seg_all = self.relation_seg_all_list[self.batch_cnt: ]
			sequence_length_seg = self.relation_seg_sequence_length_list[self.batch_cnt: ]
			sequence_length_seg_all = self.relation_seg_all_sequence_length_list[self.batch_cnt: ]
			sequence_length_question = self.question_sequence_length_list[self.batch_cnt: ]

			question_matrix = self.build_calculate_matrix_for_question(self.case_num_list[self.batch_cnt: ])
			similarity_matrix = self.build_calculate_matrix_for_similarity(self.case_num_list[self.batch_cnt: ])	

			self.batch_cnt = self.total_batch

			return relation_seg, relation_seg_all, question, \
								sequence_length_seg, sequence_length_seg_all, sequence_length_question, \
								question_matrix, similarity_matrix

	def build_calculate_matrix_for_question(self, case_num):
		matrix = []
		for i in range(len(case_num)):
			vec = [0] * len(case_num)
			vec[i] = 1
			matrix.extend([vec] * case_num[i])
		return np.asarray(matrix)

	def build_calculate_matrix_for_similarity(case_num):
		matrix = []
		height = sum(case_num)
		width = height - len(case_num)
		for i in range(height):
			matrix.append([0]*width)

		startpoint = 0
		for i in range(len(case_num)):
			for j in range(1, case_num[i]):
				matrix[startpoint][startpoint-i+j-1] = -1
				matrix[startpoint+j][startpoint-i+j-1] = 1

			startpoint += case_num[i]

		return np.asarray(matrix)


	def data_process(self, data, max_length_dict):
		question_list = []
		relation_seg_list = []
		relation_seg_all_list = []
		question_sequence_length_list = []
		relation_seg_sequence_length_list = []
		relation_seg_all_sequence_length_list = []
		case_num_list = []

		for one_data in data:
			new_one_data = []

			question = one_data[2]

			gold_relation = one_data[0]
			gold_relation_seg = gold_relation[0]
			gold_relation_seg_all = gold_relation[1]

			question_list.append(self.padding(question, max_length_dict['question_max_length']))
			question_sequence_length_list.append(len(question))

			relation_seg_list.append(self.padding(gold_relation_seg, max_length_dict['seg_max_length']))
			relation_seg_all_list.append(self.padding(gold_relation_seg_all, max_length_dict['seg_all_max_length']))
			relation_seg_sequence_length_list.append(len(gold_relation_seg))
			relation_seg_all_sequence_length_list.append(len(gold_relation_seg_all))
			

			case_num_list.append(len(one_data[1]) + 1)

			for neg_data in one_data[1]:
				neg_relation_seg = neg_data[0]
				neg_relation_seg_all = neg_data[1]

				relation_seg_list.append(self.padding(neg_relation_seg, max_length_dict['seg_max_length']))
				relation_seg_all_list.append(self.padding(neg_relation_seg_all, max_length_dict['seg_all_max_length']))
				relation_seg_sequence_length_list.append(len(neg_relation_seg))
				relation_seg_all_sequence_length_list.append(len(neg_relation_seg_all))

		
		return np.asarray(question_list), np.asarray(relation_seg_list), np.asarray(relation_seg_all_list), \
				np.asarray(question_sequence_length_list), np.asarray(relation_seg_sequence_length_list), \
				np.asarray(relation_seg_all_sequence_length_list), case_num_list




	def padding(self, data, max_length):
		new_data = []
		for i in range(max_length):
			if i < len(data):
				new_data.append(data[i])
			else:
				new_data.append(0)

		return data


def max_length(data):
	gold_seg_max_length = 0
	gold_seg_all_max_length = 0
	neg_seg_max_length = 0
	neg_seg_all_max_length = 0
	question_max_length = 0

	for one_data in data:
		gold_relation = one_data[0]
		gold_seg_max_length = max(gold_seg_max_length, len(gold_relation[0]))
		gold_seg_all_max_length = max(gold_seg_all_max_length, len(gold_relation[1]))

		for neg_data in one_data[1]:
			neg_seg_max_length = max(neg_seg_max_length, len(neg_data[0]))
			neg_seg_all_max_length = max(neg_seg_all_max_length, len(neg_data[1]))

		question_max_length = max(question_max_length, len(one_data[2]))


	return gold_seg_max_length, gold_seg_all_max_length, \
			neg_seg_max_length, neg_seg_all_max_length, \
			question_max_length


def data_static(training_data, testing_data, valid_data):
	training_gold_seg_max_length, training_gold_seg_all_max_length, \
		training_neg_seg_max_length, training_neg_seg_all_max_length, \
			training_question_max_length = max_length(training_data)

	testing_gold_seg_max_length, testing_gold_seg_all_max_length, \
		testing_neg_seg_max_length, testing_neg_seg_all_max_length, \
			testing_question_max_length = max_length(testing_data)

	valid_gold_seg_max_length, valid_gold_seg_all_max_length, \
		valid_neg_seg_max_length, valid_neg_seg_all_max_length, \
			valid_question_max_length = max_length(valid_data)

	gold_seg_max_length = max(training_gold_seg_max_length, testing_gold_seg_max_length, valid_gold_seg_max_length)
	gold_seg_all_max_length = max(training_gold_seg_all_max_length, testing_gold_seg_all_max_length, valid_gold_seg_all_max_length)
	neg_seg_max_length = max(training_neg_seg_max_length, testing_neg_seg_max_length, valid_neg_seg_max_length)
	neg_seg_all_max_length = max(training_neg_seg_all_max_length, testing_neg_seg_all_max_length, valid_neg_seg_all_max_length)
	
	seg_max_length = max(gold_seg_max_length, neg_seg_max_length)
	seg_all_max_length = max(gold_seg_all_max_length, neg_seg_all_max_length)
	question_max_length = max(training_question_max_length, testing_question_max_length, valid_question_max_length)
	
	max_length_dict = dict()
	max_length_dict["seg_max_length"] = seg_max_length
	max_length_dict["seg_all_max_length"] = seg_all_max_length
	max_length_dict["question_max_length"] = question_max_length

	return max_length_dict



