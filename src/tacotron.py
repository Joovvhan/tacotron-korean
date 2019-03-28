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
			conv_bank_outputs = conv1d_bank(prenet_outputs)

		max_pooling_outputs = tf.layers.max_pooling1d(conv_bank_outputs, pool_size=2, strides=1, padding="same")

		with tf.variable_scope('conv1d_proj_1', reuse=tf.AUTO_REUSE):
			conv1d_proj_1_outputs = conv1d_proj(max_pooling_outputs)
			normalized_conv1d_proj_1_outputs = batch_normalization(conv1d_proj_1_outputs)

		with tf.variable_scope('conv1d_proj_2', reuse=tf.AUTO_REUSE):
			conv1d_proj_2_outputs = conv1d_proj(normalized_conv1d_proj_1_outputs)
			normalized_conv1d_proj_2_outputs = batch_normalization(conv1d_proj_2_outputs)
		
		residual_outputs = prenet_outputs + normalized_conv1d_proj_2_outputs

		with tf.variable_scope('highwaynet', reuse=tf.AUTO_REUSE):
			highwaynet_outputs = highwaynet(residual_outputs)

		with tf.variable_scope('bidirectional_gru', reuse=tf.AUTO_REUSE):
			memory = bidirectional_gru(highwaynet_outputs)

		with tf.variable_scope('decoder_prenet', reuse=tf.AUTO_REUSE):
			prenet_outputs = prenet(decoder_inputs, True)

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

def conv1d_proj(inputs):

	params = {"inputs":inputs, 
				"filters":hp.num_conv1d_proj_filter, 
				"kernel_size":hp.size_conv1d_proj_filter,
				"dilation_rate":1, 
				"padding":"SAME", 
				"activation":None, 
				"use_bias":False
				}

	result = tf.layers.conv1d(**params)

	return result

def batch_normalization(inputs):

	params = {"inputs":inputs,
				"center":True,
				"scale":True,
				"updates_collections":None,
				"is_training":True,
				"scope":"conv1d_1",
				"fused":True
				}

	result = tf.contrib.layers.batch_norm(**params)

	return result

def highwaynet(inputs):

	for i in range(hp.num_highwaynet_blocks):

		scope = "highwaynet_{:d}".format(i)
		
		with tf.variable_scope(scope):

			if i == 0:
				highwaynet_input = inputs
			else:
				highwaynet_input = highwaynet_output

			H = tf.layers.dense(highwaynet_input, 
								units=hp.num_highwaynet_units, 
								activation=tf.nn.relu, 
								name="dense1", 
								reuse=tf.AUTO_REUSE)

			T = tf.layers.dense(highwaynet_input, 
								units=hp.num_highwaynet_units, 
								activation=tf.nn.sigmoid,
								bias_initializer=tf.constant_initializer(-1.0), 
								name="dense2", 
								reuse=tf.AUTO_REUSE)

			highwaynet_output = H * T + highwaynet_input*(1.0 - T)

	return highwaynet_output

def bidirectional_gru(inputs):

	cell_fw = tf.contrib.rnn.GRUCell(hp.num_gru_units)
	cell_bw = tf.contrib.rnn.GRUCell(hp.num_gru_units)

	rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32)
	results = tf.concat(rnn_outputs, 2)  
	return results 

def attention(inputs)

	attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(hp.num_attention_units, inputs)
	rnn_cell = tf.contrib.rnn.GRUCell(hp.num_gru_units)
	cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(rnn_cell,
															  attention_mechanism,
															  hp.num_attention_units,
															  alignment_history=True,
															  output_attention=False)
	
	dec, state = tf.nn.dynamic_rnn(cell_with_attention, inputs, dtype=tf.float32)