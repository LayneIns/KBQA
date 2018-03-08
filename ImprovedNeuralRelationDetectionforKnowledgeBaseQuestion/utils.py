# -*- coding: utf-8 -*-
import re
import sys
import numpy as np 


def readRelations(datapath):
	relation_list = []
	with open(datapath) as fin:
		for line in fin:
			new_line = line.decode("utf-8").strip()
			if new_line.strip():
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


def wordStatForRelation(relation_list_seg, relation_list_seg_all, training_data):
	word_dict = dict()
	word_dict["#UNK#"] = len(word_dict)

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
		

	# for relation in relation_list_seg:
	# 	for word in relation:
	# 		if word_dict.get(word, -1) == -1:
	# 			word_dict[word] = len(word_dict)
	# for relation in relation_list_seg_all:
	# 	for word in relation:
	# 		if word_dict.get(word, -1) == -1:
	# 			word_dict[word] = len(word_dict)

	print "There are", len(word_dict), "words in all relations."
	return word_dict


def readData(datapath):
	data_list = []
	with open(datapath) as fin:
		for line in fin:
			new_line = line.decode("utf-8").strip()
			if new_line.strip():
				one_data = new_line.split("\t")

			gold_relation = int(one_data[0])
			neg_relation = one_data[1].split()
			if neg_relation[0] == "noNegativeAnswer":
				neg_relation = []
			else:
				neg_relation = [int(num) for num in neg_relation if num.strip()]
			question = one_data[2].split()
			one_data = []
			one_data.append(gold_relation)
			one_data.append(neg_relation)
			one_data.append(question)
			data_list.append(one_data)
	# print data_list[-1]
	print "There are", len(data_list), "cases."
	return data_list		


def wordStatForQuestion(data_list):
	word_dict = dict()
	word_dict["#head_entity#"] = len(word_dict)
	word_dict["#UNK#"] = len(word_dict)
	for one_data in data_list:
		question = one_data[2]
		for word in question:
			if word_dict.get(word, -1) == -1:
				word_dict[word] = len(word_dict)

	print "There are", len(word_dict), "words in the training data."
	return word_dict


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
			new_gold_relation_seg.append(relation_words.get(word, relation_words.get("#UNK#")))
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
		
		# print new_one_data
		# raw_input()

	return new_data	


def question_words_embedding(question_words, embedding_filepath):
	all_word_embedding = dict()
	with open(embedding_filepath) as fin:
		line_cnt = 0
		for line in fin:
			if line_cnt % 1000 == 0:
				sys.stdout.write(" " * 30 + "\r")
				sys.stdout.flush()
				sys.stdout.write("Now reading line " + str(line_cnt))
				sys.stdout.flush()
			line_cnt += 1

			if line.strip():
				seg_res = line.split(" ")
				seg_res = [word.strip() for word in seg_res if word.strip()]

				key = seg_res[0]
				value = [float(word) for word in seg_res[1:]]
				if len(value) != 300:
					print "Wrong vector for word embedding"
				all_word_embedding[key] = value

	print "\nRead all word embeddings."

	reverse_question_words = dict()
	for key, value in question_words.items():
		reverse_question_words[str(value)] = key

	print "Reversed word index built."

	embedding_matrix = []
	for i in range(len(reverse_question_words)):
		sys.stdout.write(" " * 30 + "\r")
		sys.stdout.flush()
		sys.stdout.write("Buiding word " + str(i) + "/" + str(len(reverse_question_words)))	
		sys.stdout.flush()

		if i == 0 or i == 1:
			value = np.random.uniform(low=-0.5, high=0.5, size=(300,)).tolist()
			embedding_matrix.append(value)
			continue
		i_str = str(i)
		key = reverse_question_words[i_str]
		value = all_word_embedding.get(key, -1)
		if value == -1:
			value = np.random.uniform(low=-0.5, high=0.5, size=(300,)).tolist()
		embedding_matrix.append(value)

	embedding_matrix = np.asarray(embedding_matrix)
	print "\nThere size of word_embedding matrix is", embedding_matrix.shape
	
	return embedding_matrix







