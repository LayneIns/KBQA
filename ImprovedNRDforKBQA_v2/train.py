# -*- coding:utf-8 -*-

import tensorflow as tf 
from model import HrbilstmModel
import numpy as np 

def train_model(arg_config, training_data_mgr, testing_data_mgr, valid_data_mgr):
	model = HrbilstmModel
	saver = tf.train.Saver(max_to_keep=1)
	
	with tf.Session() as sess:
		
