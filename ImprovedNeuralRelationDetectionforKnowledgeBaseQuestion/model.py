import tensorflow as tf 


# Model from Improved Neural Relation Detection 
# for Knowledge Base Question Answering
class HrbilstmModel:

	def __init__(self, arg_config):
		self.relation_seg_input = tf.placeholder(tf.int64, [None, None])
		self.relation_seg_all_input = tf.placeholder(tf.int64, [None, None])

		self.question_input = tf.placeholder(tf.float32, [None, None, arg_config["word_embedding_size"]])

		with tf.variable_scope("relation_embedding_layer"):
			W = tf.get_variable("W", [arg_config["vocab_size"], arg_config["relation_embedding_size"]], initializer=tf.random_normal_initializer(mean=0, stddev=1))
			self.relation_seg_embedding = tf.nn.embedding_lookup(W, self.relation_seg_input)
			self.relation_seg_all_embedding = tf.nn.embedding_lookup(W, self.relation_seg_all_input)




	def 

		