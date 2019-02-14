# -*- coding: utf-8 -*-

class Hyperparams:

	vocab = " ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ.,?!"
	data_dir = 'D:/korean-single-speaker-speech-dataset/kss'
	mels_dir = 'D:/korean-single-speaker-speech-dataset/kss/mels'
	mags_dir = 'D:/korean-single-speaker-speech-dataset/kss/mags'
	transcript_pos = 'D:/korean-single-speaker-speech-dataset/kss/transcript.txt'
	nsc_sec = 0.1
	nov_sec = 0.05
	n_mels = 80
	eps = 1e-5
	max_db = 100
	embed_size = 256
	num_k_filter = embed_size//2
	K = 16