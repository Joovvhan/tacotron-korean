# -*- coding: utf-8 -*-

class Hyperparams:

	vocab = " ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ.,?!"
	data_dir = 'D:/korean-single-speaker-speech-dataset/kss'
	mels_dir = 'D:/korean-single-speaker-speech-dataset/kss/mels'
	mags_dir = 'D:/korean-single-speaker-speech-dataset/kss/mags'
	transcript_pos = 'D:/korean-single-speaker-speech-dataset/kss/transcript.txt'
	fs = 44100/2
	nsc_sec = 0.1
	nov_sec = 0.05
	n_mels = 80
	# db_limit = -60
	db_limit = -80
	offset = 2
	dropout_rate = 0.5

	queue_size = 8
	
	r = 5
	
	lr = 0.001

	# batch_size = 32
	# group_size = 8

	batch_size = 4
	group_size = 2
	
	eps = 1e-8
	max_db = 160 # 20 * log10(eps) == -160
	embed_size = 256
	num_prenet_node_1 = 256
	num_prenet_node_2 = 128
	num_k_filter = embed_size//2
	num_conv1d_proj_filter = embed_size//2
	size_conv1d_proj_filter = 3
	num_highwaynet_blocks = 4
	num_highwaynet_units = embed_size//2
	num_gru_units = embed_size//2
	
	num_attention_units = embed_size//2
	
	K = 16
	