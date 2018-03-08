# -*- coding:utf-8 -*-

import sys
from utils import readRelations, readData, wordStatForRelation, \
					wordStatForQuestion, convert_data, \
					question_words_embedding

from data import data_static


if __name__ == "__main__":

	print "Start to read relations"
	relation_list_seg, relation_list_seg_all = \
						readRelations("KBQA_data/sq_relations/relation.2M.list")



	print "\n"
	print "Start to read training data"
	training_data = readData("KBQA_data/sq_relations/train.replace_ne.withpool")
	print "Start to read testing data"
	testing_data = readData("KBQA_data/sq_relations/test.replace_ne.withpool")
	print "Start to read validation data"
	valid_data = readData("KBQA_data/sq_relations/valid.replace_ne.withpool")


	print "\n"
	print "Start to get word dictionary for question and relation"
	question_words = wordStatForQuestion(training_data)
	relation_words = wordStatForRelation(relation_list_seg, relation_list_seg_all, training_data)
	

	print "\n"
	print "Start to convert data to vectors"
	training_data_conv = convert_data(question_words, relation_words, relation_list_seg, relation_list_seg_all, training_data)
	print "\nThere are", len(training_data_conv), "in the training data"
	testing_data_conv = convert_data(question_words, relation_words, relation_list_seg, relation_list_seg_all, testing_data)
	print "\nThere are", len(testing_data_conv), "in the testing data"
	valid_data_conv = convert_data(question_words, relation_words, relation_list_seg, relation_list_seg_all, valid_data)
	print "\nThere are", len(valid_data_conv), "in the valid data"
	

	print "\n"
	print "Start to get word word embedding matrix"
	word_embedding_matrix = question_words_embedding(question_words, "glove/glove.6B.300d.txt")
	# print word_embedding_matrix[0:3]
	# print "\n"
	# print word_embedding_matrix[-3:-1]


	print "\nStart to calculate the max length for sequence length"
	max_length_dict = data_static(training_data_conv, testing_data_conv, valid_data_conv)

	print max_length_dict



	





