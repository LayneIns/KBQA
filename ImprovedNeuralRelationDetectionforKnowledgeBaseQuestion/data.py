import numpy as np 


class dataMgr():

	def __init__(self, data):
		all_data = self.data_process(data)

	def data_process(self, data):
		new_data = []
		for one_data in data:
			new_one_data = []
			gold_relation = one_data[0]
			gold_relation_seg = gold_relation_seg[0]
			gold_relation_seg_all = gold_relation_seg[1]




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
	question_max_length = max(training_question_max_length, testing_question_max_length, valid_question_max_length)
	
	max_length_dict = dict()
	max_length_dict["gold_seg_max_length"] = gold_seg_max_length
	max_length_dict["gold_seg_all_max_length"] = gold_seg_all_max_length
	max_length_dict["neg_seg_max_length"] = neg_seg_max_length
	max_length_dict["neg_seg_all_max_length"] = neg_seg_all_max_length
	max_length_dict["question_max_length"] = question_max_length

	return max_length_dict



