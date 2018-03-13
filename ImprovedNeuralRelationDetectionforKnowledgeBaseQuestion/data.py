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
		self.all_data = self.data_process(data, max_length_dict)
		self.total_batch = len(self.all_data)
		self.batch_cnt = 0
	
	def initialize_batch_cnt(self):
		self.batch_cnt = 0

	def next_batch(self):
		if self.batch_cnt != self.total_batch:
			one_batch = self.all_data[self.batch_cnt]

			relation_seg = []
			relation_seg_all = []
			question = []
			sequence_length_seg = []
			sequence_length_seg_all = []
			sequence_length_question = []
			label_sequence = []


			gold_case = one_batch[0]
			gold_relation_seg = gold_case[0]
			gold_relation_seg_all = gold_case[1]
			gold_question = gold_case[2]
			gold_relation_seg_length = gold_case[3]
			gold_relation_seg_all_length = gold_case[4]
			gold_question_length = gold_case[5]

			relation_seg.append(gold_relation_seg)
			relation_seg_all.append(gold_relation_seg_all)
			question.append(gold_question)
			sequence_length_seg.append(gold_relation_seg_length)
			sequence_length_seg_all.append(gold_relation_seg_all_length)
			sequence_length_question.append(gold_question_length)

			label_sequence.append(-1)

			for neg_case in one_batch[1:]:
				neg_relation_seg = neg_case[0]
				neg_relation_seg_all = neg_case[1]
				neg_question = neg_case[2]
				neg_realtion_seg_length = neg_case[3]
				neg_relation_seg_all_length = neg_case[4]
				neg_question_length = neg_case[5]

				relation_seg.append(neg_relation_seg)
				relation_seg_all.append(neg_relation_seg_all)
				question.append(neg_question)
				sequence_length_seg.append(neg_realtion_seg_length)
				sequence_length_seg_all.append(neg_relation_seg_all_length)
				sequence_length_question.append(neg_question_length)

				label_sequence.append(1)

			self.batch_cnt += 1

			return relation_seg, relation_seg_all, question, \
								sequence_length_seg, sequence_length_seg_all, sequence_length_question, \
								label_sequence


	def data_process(self, data, max_length_dict):
		new_data = []
		for one_data in data:
			new_one_data = []

			question = one_data[2]

			gold_relation = one_data[0]
			gold_relation_seg = gold_relation_seg[0]
			gold_relation_seg_all = gold_relation_seg[1]
			
			gold_case = []
			gold_case.append(self.padding(gold_relation_seg, max_length_dict['seg_max_length']))
			gold_case.append(self.padding(gold_relation_seg_all, max_length_dict['seg_all_max_length']))
			gold_case.append(self.padding(question, max_length_dict['question_max_length']))
			gold_case.append(len(gold_relation_seg))
			gold_case.append(len(gold_relation_seg_all))
			gold_case.append(len(question))
			
			new_one_data.append(gold_case)

			for neg_data in one_data[1]:
				neg_relation_seg = neg_data[0]
				neg_relation_seg_all = neg_data[1]
				
				neg_case = []
				neg_case.append(self.padding(neg_relation_seg, max_length_dict['seg_max_length']))
				neg_case.append(self.padding(neg_relation_seg_all, max_length_dict['seg_all_max_length']))
				neg_case.append(self.padding(question, max_length_dict['question_max_length']))
				neg_case.append(len(neg_relation_seg))
				neg_case.append(len(neg_relation_seg_all))
				neg_case.append(len(question))

				new_one_data.append(neg_case)

			new_data.append(new_one_data)

		return new_data

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



