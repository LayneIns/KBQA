# coding: utf-8

import sys
from utils import readRelations, readData, wordStatForQuestion, wordStatForRelation, \
		convert_data
from data import data_static, argConfig, dataMgr
from train import train_model


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
	print "\nThere are", len(training_data_conv), "in the training data"
	testing_data_conv = convert_data(question_words, relation_words, relation_list_seg, relation_list_seg_all, testing_data)
	print "\nThere are", len(testing_data_conv), "in the testing data"
	valid_data_conv = convert_data(question_words, relation_words, relation_list_seg, relation_list_seg_all, valid_data)
	print "\nThere are", len(valid_data_conv), "in the valid data"
	print "\n"
	
	print "Start to calculate the max length for sequence length..."
	max_length_dict = data_static(training_data_conv, testing_data_conv, valid_data_conv)
	print max_length_dict
	print "\n"
	
	arg_config = argConfig(max_length_dict, len(relation_words), len(question_words))
	
	print "Start to building data manager..."
	training_data_mgr = dataMgr(training_data_conv, max_length_dict)
	print "training_data_mgr: total_batch:", training_data_mgr.total_batch
	testing_data_mgr = dataMgr(testing_data_conv, max_length_dict)
	print "testing_data_mgr: total_batch:", testing_data_mgr.total_batch
	valid_data_mgr = dataMgr(valid_data_conv, max_length_dict)
	print "valid_data_mgr: total_batch:", valid_data_mgr.total_batch
	print "\n"
	
	print "Start to train..."
	train_model(arg_config, training_data_mgr, testing_data_mgr, valid_data_mgr)
	
	
	
	
	
	
	
	
	
	
	
	