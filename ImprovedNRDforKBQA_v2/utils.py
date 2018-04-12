# coding: utf-8

import re
import sys
import numpy as np

# Get all the relations, and return two list:
# assuming that we have a relation "/martial_arts/martial_artist/martial_art"
# relation_list_seg: ["martial_arts", "martial_artist", "martial_art"]
# relation_list_seg_all: ["martial", "arts", "martial", "artist", "martial", "art"]
def readRelations(datapath):
	relation_list = []
	with open(datapath) as fin:
		for line in fin:
			new_line = line.decode("utf-8").strip()
			if new_line:
				relation_list.append(new_line)
	
	print "There are", len(relation_list), "relations in the dataset."
	
	relation_list_seg = []
	relation_list_seg_all = []
	for relation in relation_list:
		relation_seg = relation.split("/")
		relation_seg = [seg_part.strip() for seg_part in relation_seg if seg_part.strip()]
		relation_list_seg.append(relation_seg)
		relation_seg_all = []
		for word in relation_seg:
			relation_seg_all.extend(word.split("_"))
		relation_list_seg_all.append(relation_seg_all)
		
	return relation_list_seg, relation_list_seg_all


# Get all the data for training, testing and validating
# Return a list data_list, and each element in that list is a list formed as [gold_relation, negative_relation, question], 
# here, gold_relation is a number, negative_relation is a list composing of the numbers of the negative samples, and question is a list composing of the words of the sentence
def readData(datapath, train=True):
	data_list = []
	with open(datapath) as fin:
		for line in fin:
			new_line = line.decode("utf-8").strip()
			if new_line:
				one_data = new_line.split("\t")
			else:
				continue
			
			gold_relation = int(one_data[0])
			if one_data[1] == "noNegativeAnswer":
				neg_relation = []
			else:
				neg_relation = [int(num) for num in one_data[1].split()]
			if train == False:
				neg_relation.remove(gold_relation)
			question = one_data[2].split()
			one_data = []
			one_data.append(gold_relation)
			one_data.append(neg_relation)
			one_data.append(question)
			data_list.append(one_data)
	print "There are", len(data_list), "cases."
	return data_list
	

# Get the word dictionary for the questions in the training data
# Return a python dictionary word_dict, noticing that word_dict['#UNK#']=0, and word_dict['#head_entity#']=1
def wordStatForQuestion(data_list):
	word_dict = dict()
	word_dict['#UNK#'] = len(word_dict)
	word_dict["#head_entity#"] = len(word_dict)
	for one_data in data_list:
		question = one_data[2]
		for word in question:
			if word_dict.get(word, -1) == -1:
				word_dict[word] = len(word_dict)
	print "There are", len(word_dict), "words in the training data questions."
	return word_dict


# Get the word dictionary for the relations in the training data
#Return a python dictionary word_dict, noticing that word_dict['#UNK#']=0
def wordStatForRelation(relation_list_seg, relation_list_seg_all, training_data):
	word_dict = dict()
	word_dict['#UNK#'] = len(word_dict)
	
	for one_data in training_data:
		gold_relation = one_data[0]
		neg_relation = one_data[1]
		gold_relation_seg = relation_list_seg[gold_relation - 1]
		gold_relation_seg_all = relation_list_seg_all[gold_relation - 1]
		neg_relation_seg = []
		neg_relation_seg_all = []
		for one_neg in neg_relation:
			neg_relation_seg.append(relation_list_seg[one_neg - 1])
			neg_relation_seg_all.append(relation_list_seg_all[one_neg - 1])
		
		for word in gold_relation_seg:
			if word_dict.get(word, -1) == -1:
				word_dict[word] = len(word_dict)
		for word in gold_relation_seg_all:
			if word_dict.get(word, -1) == -1:
				word_dict[word] = len(word_dict)
		
		for relation in neg_relation_seg:
			for word in relation:
				if word_dict.get(word, -1) == -1:
					word_dict[word] = len(word_dict)
		for relation in neg_relation_seg_all:
			for word in relation:
				if word_dict.get(word, -1) == -1:
					word_dict[word] = len(word_dict)
	
	print "There are", len(word_dict), "words in all relations."
	return word_dict
	

# Convert the words in the relations(seg or not seg) and questions into their indexes
# Return a list new_data, each element of new_data is a list [gold_relation, neg_relation, question]
# Noticing that the neg_relation here is a list because there might be several candidate negative relations
def convert_data(question_words, relation_words, relation_list_seg, relation_list_seg_all, data):
	new_data = []
	data_cnt = 0
	for one_data in data:
		sys.stdout.write(" " * 30 + "\r")
		sys.stdout.flush()
		sys.stdout.write("Now converting " + str(data_cnt) + "/" + str(len(data)))
		sys.stdout.flush()
		data_cnt += 1
		
		new_one_data = []
		
		gold_relation = one_data[0]
		neg_relation = one_data[1]
		question = one_data[2]
		gold_relation_seg = relation_list_seg[gold_relation - 1]
		gold_relation_seg_all = relation_list_seg_all[gold_relation - 1]
		
		new_gold_relation_seg = []
		for word in gold_relation_seg:
			new_gold_relation_seg.append(relation_words.get(word, relation_words.get("#UNK")))
		new_gold_relation_seg_all = []
		for word in gold_relation_seg_all:
			new_gold_relation_seg_all.append(relation_words.get(word, relation_words.get("#UNK#")))
		
		new_one_data.append([new_gold_relation_seg, new_gold_relation_seg_all])
		
		new_neg_relation = []
		for one_neg_relation in neg_relation:
			one_neg_relation_seg = relation_list_seg[one_neg_relation - 1]
			one_neg_relation_seg_all = relation_list_seg_all[one_neg_relation - 1]
			
			new_one_neg_relation_seg = []
			for word in one_neg_relation_seg:
				new_one_neg_relation_seg.append(relation_words.get(word, relation_words.get("#UNK#")))
			new_one_neg_relation_seg_all = []
			for word in one_neg_relation_seg_all:
				new_one_neg_relation_seg_all.append(relation_words.get(word, relation_words.get("#UNK#")))
			new_neg_relation.append([new_one_neg_relation_seg, new_one_neg_relation_seg_all])
		
		new_one_data.append(new_neg_relation)
		
		new_question = []
		for word in question:
			new_question.append(question_words.get(word, question_words.get("#UNK#")))
		
		new_one_data.append(new_question)
		
		new_data.append(new_one_data)
	
	return new_data

		

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	