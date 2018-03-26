# coding: utf-8

import sys
from utils import readRelations, readData, wordStatForQuestion, wordStatForRelation, \
		convert_data

if __name__ == "__main__":
	print "Start to read relations..."
	relation_list_seg, relation_list_seg_all = \
				readRelations("KBQA_data/sq_relations/relation.2M.list")
	print "\n"
	
	print "Start to read training data..."
	training_data = readData("KBQA_data/sq_relations/train.replace_ne.withpool")
	print "Start to read testing data..."
	testing_data = readData("KBQA_data/sq_relations/test.replace_ne.withpool")
	print "Start to read validation data"
	valid_data = readData("KBQA_data/sq_relations/valid.replace_ne.withpool")
	print "\n"
	
	print "start to get word dictionary for questions and relations"
	question_words = wordStatForQuestion(training_data)
	relation_words = wordStatForRelation(relation_list_seg, relation_list_seg_all, training_data)
	print "\n"
	
	print "Start to convert data to vectors..."
	training_data_conv = convert_data(question_words, relation_words, relation_list_seg, relation_list_seg_all, training_data)
	
	
	
	
	
	
	