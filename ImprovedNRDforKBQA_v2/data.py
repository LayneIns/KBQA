# -*- coding:utf-8 -*-

import numpy as np 

# Model setup arguments
class argConfig:
	
	def __init__(self, max_length_dict, relation_vocab_size, question_vocab_size):
		self.relation_embedding_size = 200
		self.question_embedding_size = 200
		
		self.relation_lstm_size = 300
		self.question_lstm_size = 300
		
		self.relation_vocab_size = relation_vocab_size
		self.question_vocab_size = question_vocab_size
		
		self.max_length_dict = max_length_dict
		
		self.batch_size = 15
		self.epoch_num = 10
		self.gamma = 0.1
		self.learning_rate = 0.5
		
		'''
		self.relation_ksize = [1, max_length_dict['seg_max_length'] + max_length_dict['seg_all_max_length'], 1, 1]
		self.question_ksize = [1, max_length_dict['question_max_length'], 1, 1]
		'''


class dataMgr:
	
	def __init__(self, data, max_length_dict):
		self.max_length_dict = max_length_dict
		
		self.padding_data, self.case_num_list, \
				self.sequence_list = self.data_process(data, max_length_dict)
		self.total_batch = len(self.case_num_list)
		self.batch_cnt = 0
	
	def initialize_batch_cnt(self):
		self.batch_cnt = 0
	
	def build_data(self, data, sequence_length_list):
		question_data = []
		relation_seg = []
		relation_seg_all = []
		seg_sequence_list = []
		seg_all_sequence_list = []
		question_sequence_list = []
		for one_data in data:
			relation_seg.extend(one_data[0])
			relation_seg_all.extend(one_data[1])
			question_data.extend(one_data[2])
		for one_sequence_length in sequence_length_list:
			seg_sequence_list.extend(one_sequence_length[0])
			seg_all_sequence_list.extend(one_sequence_length[1])
			question_sequence_list.extend(one_sequence_length[2])
			
		return np.asarray(relation_seg, dtype=np.int32), np.asarray(relation_seg_all, dtype=np.int32), np.asarray(question_data, dtype=np.int32), np.asarray(seg_sequence_list, dtype=np.int64), np.asarray(seg_all_sequence_list, dtype=np.int64), np.asarray(question_sequence_list, dtype=np.int64)
		
	
	def next_batch(self, batch_size):
		print "Batch_cnt:", self.batch_cnt
		if self.batch_cnt <= self.total_batch - batch_size:
			ret_data = self.padding_data[self.batch_cnt: self.batch_cnt + batch_size]
			ret_case_num = self.case_num_list[self.batch_cnt: self.batch_cnt + batch_size]
			ret_sequence_length = self.sequence_list[self.batch_cnt: self.batch_cnt + batch_size]
			
			relation_seg, relation_seg_all, question, seg_sequence_list, seg_all_sequence_list, question_sequence_list = self.build_data(ret_data, ret_sequence_length)
			self.batch_cnt += batch_size
		elif self.batch_cnt < self.total_batch:
			ret_data = self.padding_data[self.batch_cnt: ]
			ret_case_num = self.case_num_list[self.batch_cnt: ]
			ret_sequence_length = self.sequence_list[self.batch_cnt: ]
			
			relation_seg, relation_seg_all, question, seg_sequence_list, seg_all_sequence_list, question_sequence_list = self.build_data(ret_data, ret_sequence_length)
			self.batch_cnt = self.total_batch
		else:
			return None, None, None, None, None, None
		
		return relation_seg, relation_seg_all, question, seg_sequence_list, seg_all_sequence_list, question_sequence_list
		
	def data_process(self, data, max_length_dict):
		padding_data = []
		case_num_list = []
		sequence_list = []
		for one_data in data:			
			seg_data = []
			seg_all_data = []
			question_list = []
			
			seg_sequence_length = []
			seg_all_sequence_length = []
			question_sequence_length = []
			
			gold_relation = one_data[0]
			gold_relation_seg = gold_relation[0]
			gold_relation_seg_all = gold_relation[1]
			
			seg_sequence_length.append(len(gold_relation_seg))
			seg_all_sequence_length.append(len(gold_relation_seg_all))
			
			seg_data.append(self.padding(gold_relation_seg, max_length_dict['seg_max_length']))
			seg_all_data.append(self.padding(gold_relation_seg_all, max_length_dict['seg_all_max_length']))
			
			for neg_data in one_data[1]:
				neg_relation_seg = neg_data[0]
				neg_relation_seg_all = neg_data[1]
				
				seg_sequence_length.append(len(neg_relation_seg))
				seg_all_sequence_length.append(len(neg_relation_seg_all))
			
				seg_data.append(self.padding(neg_relation_seg, max_length_dict['seg_max_length']))
				seg_all_data.append(self.padding(neg_relation_seg_all, max_length_dict['seg_all_max_length']))
			
			if len(seg_data) != len(seg_all_data):
				print "Wrong length for data alignment."
			
			case_num_list.append(len(seg_data))
			
			question = self.padding(one_data[2], max_length_dict['question_max_length'])
			for i in range(len(seg_data)):
				question_sequence_length.append(len(question))
				question_list.append(question)
			
			new_one_data = [seg_data, seg_all_data, question_list]
			one_sequence_length = [seg_sequence_length, seg_all_sequence_length, question_sequence_length]
			
			padding_data.append(new_one_data)
			sequence_list.append(one_sequence_length)
			
			
		return padding_data, case_num_list, sequence_list
	
	def padding(self, data, max_length):
		new_data = []
		for i in range(max_length):
			if i < len(data):
				new_data.append(data[i])
			else:
				new_data.append(0)
		return new_data

		
		
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


# Get the max length for relation_seg, relation_seg_all, question.
# Return a python dictionary, containing three elements: "seg_max_length", "seg_all_max_length", "question_max_length"
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
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	