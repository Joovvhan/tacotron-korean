import threading
import os
import librosa
import random
from hyperparams import Hyperparams as hp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class DataFeeder(threading.Thread):

	def __init__(self, metadata):
		super(DataFeeder, self).__init__()
		self._metadata = metadata
		self._cur = 0

		self._placeholders = [
			tf.placeholder(tf.int32, [None, None], name='text'),
			tf.placeholder(tf.float32, [None, None, None], name='mel'),
			tf.placeholder(tf.float32, [None, None, None], name='mag'),
			tf.placeholder(tf.int32, [None], name='text_length'),
		]

	def start_thread(self):
		self.start()

	def run(self):
		print(os.getpid())

	def _load_wav(self, fname):
		fpath = os.path.join(hp.data_dir, fname)
		wav, fs = librosa.core.load(fpath, mono=True)

		# plt.plot(wav)
		# plt.show()

		return wav, fs

	def mel_spectrogram(self, wav, nsc, nov, fs):
		S = librosa.feature.melspectrogram(y=wav, sr=fs, n_fft=nsc, hop_length=nov, power=1.0, n_mels = hp.n_mels)
		S = S/(nsc)
		# plt.imshow(S)
		# plt.colorbar()
		# plt.show()

		dbS = 20 * np.log10(np.maximum(S, hp.eps))

		# plt.imshow(dbS)
		# plt.colorbar()
		# plt.show()

		return dbS

	def stft(self, wav, nsc, nov):
		S = librosa.core.stft(wav, n_fft=nsc, hop_length=nov)
		S = S/nsc
		Sxx = abs(S)

		# print(S)

		# print(Sxx)

		return Sxx 

	def log_stft(self, wav, nsc, nov):
		S = librosa.core.stft(wav, n_fft=nsc, hop_length=nov)
		S = S/nsc
		Sxx = abs(S)
		dbS = 20 * np.log10(np.maximum(Sxx, hp.eps))
		return dbS

	def to_dB(self, data):
		dbS = 20 * np.log10(np.maximum(data, hp.eps))
		return dbS

	def get_active_index(self, coef):

		# plt.plot(coef)
		# plt.show()

		activated_index = np.where(coef > hp.db_limit)[0]
		valid_start = activated_index[0]
		valid_last = activated_index[-1] + 1

		# print('{:d}:{:d}'.format(valid_start, valid_last))

		if (valid_last >= len(coef)):
			valid_last = len(coef) - 1

		return valid_start, valid_last

	def _load_next(self):
		if self._cur >= len(self._metadata):
			self._cur = 0
			random.shuffle(self._metadata)
		
		meta = self._metadata[self._cur]
		self._cur += 1

		fname = meta[0]
		text = meta[1]

		wav, fs = self._load_wav(fname)

		nsc = np.int(fs * hp.nsc_sec)
		nov = np.int(fs * hp.nov_sec)

		# plt.plot(wav)
		# plt.show()


		# print(nsc)

		# print(nov)
		specgram = self.stft(wav, nsc, nov)

		# plt.imshow(specgram)
		# plt.colorbar()
		# plt.show()

		stft_coef = np.mean(specgram, axis=0)
		stft_coef_dB = self.to_dB(stft_coef)
		first, last = self.get_active_index(stft_coef_dB)

		# plt.plot(stft_coef)
		# plt.show()

		mel = self.mel_spectrogram(wav, nsc, nov, fs)
		mag = self.to_dB(specgram)

		active_mel = mel[:, first:last]
		active_mag = mag[:, first:last]

		normalized_mel = (active_mel + hp.max_db)/hp.max_db
		normalized_mag = (active_mag + hp.max_db)/hp.max_db



		return text, normalized_mel, normalized_mag, normalized_mag.shape[1]

	def _load_next_group(self):

		n = hp.batch_size
		m = hp.group_size

		group = [self._load_next() for i in range(n * m)]

		group.sort(key=lambda x: x[-1])

		batches = [group[i:i+n] for i in range(0, len(group), n)]

		random.shuffle(batches)

		# for batch in batches:
		# 	feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch)))

		# return batches
		return self._prepare_batch(batches[0])

	def _prepare_batch(self, batch):
		random.shuffle(batch)

		texts = self._prepare_batch_texts([x[0] for x in batch])
		mels = self._prepare_batch_spectrograms([x[1] for x in batch])
		mags = self._prepare_batch_spectrograms([x[2] for x in batch])
		text_lengths = np.asarray([x[3] for x in batch], dtype=np.int32)

		return (texts, mels, mags, text_lengths)


	def _prepare_batch_texts(self, texts):
		max_len = max(map(lambda x: len(x), texts))
		batch_texts = np.stack(list(map(lambda x: self._prepare_text_padding(x, max_len), texts)))
		return batch_texts

	def _prepare_batch_spectrograms(self, specgrams):
		max_len = max(map(lambda x: x.shape[1], specgrams))
		padded_len = next_multiple_of_r(max_len, hp.r)

		batch_specgrams = np.stack(list(map(lambda x: self._prepare_spectrogram_padding(x, padded_len), specgrams)))
		return batch_specgrams

	def _prepare_text_padding(self, unpadded, common_len):
		assert unpadded.ndim == 1, "Target of padding operation is not 1 dimensional, but {}".format(unpadded.ndim)
		pad_len = common_len - len(unpadded)

		return np.pad(unpadded, (0, pad_len), mode='constant', constant_values=0)

	def _prepare_spectrogram_padding(self, unpadded, common_len):
		assert unpadded.ndim == 2, "Target of padding operation is not 2 dimensional, but {}".format(unpadded.ndim)
		pad_len = common_len - unpadded.shape[1]
		padded = np.pad(unpadded, [(0, 0), (0, pad_len)], mode='constant', constant_values=np.min(unpadded))
		# print('original_length: {:d}, target_len: {:d}, pad_len: {:d}, padded_len: {:d}'.format(unpadded.shape[1], common_len, pad_len, padded.shape[1]))
		# plt.imshow(padded, origin='reverse')
		# plt.show()
		return padded

def next_multiple_of_r(num, r):
	return np.int(r * np.ceil(num/r))
