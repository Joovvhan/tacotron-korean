import tensorflow as tf

from hyperparams import Hyperparams as hp

class Tacotron():
	def __init__(self, texts, mels, mags, text_lengths):
		print(texts)
		print(mels)
		print(mags)
		print(text_lengths)

		batch_size = tf.shape(texts[0])

		with tf.variable_scope('embedding_layer', reuse=tf.AUTO_REUSE):
			embedding_table = tf.get_variable('embedding',
												[len(hp.vocab), hp.embed_size],
												dtype=tf.float32,
												initializer=tf.truncated_normal_initializer(stddev=0.5))
			
			embedded_inputs = tf.nn.embedding_lookup(embedding_table, texts)

		with tf.variable_scope('prenet', reuse=tf.AUTO_REUSE):
			prenet_outputs = prenet(embedded_inputs, True)

		with tf.variable_scope('conv1d_bank', reuse=tf.AUTO_REUSE):
			self.conv_bank_outputs = conv1d_bank(prenet_outputs)


		print(self.conv_bank_outputs)

def prenet(inputs, training):

	x = inputs

	for i, num_node in enumerate(hp.num_prenet_nodes):
		dense = tf.layers.dense(x, units=num_node, activation=tf.nn.relu, name='dense_{:d}'.format(i))
		x = tf.layers.dropout(dense, rate=hp.dropout_rate, training=training, name='dropout_{:d}'.format(i))

	return x

def conv1d_bank(inputs):
	
	for k in range(1, hp.K + 1):
		with tf.variable_scope("filter_num_{}".format(k), reuse=tf.AUTO_REUSE):
			params = {"inputs":inputs, 
					  "filters":hp.num_k_filter, 
					  "kernel_size":k,
					  "dilation_rate":1, 
					  "padding":"SAME", 
					  "activation":None, 
					  "use_bias":False, 
					  "reuse":tf.AUTO_REUSE}

			conv_outputs = tf.layers.conv1d(**params)

		if k == 1:
			conv_bank_outputs = conv_outputs
		else:
			conv_bank_outputs = tf.concat((conv_bank_outputs, conv_outputs), axis=2)

	return conv_bank_outputs
