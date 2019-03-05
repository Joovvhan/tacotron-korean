import threading
import os
import librosa
import random
from hyperparams import Hyperparams as hp
import numpy as np
import matplotlib.pyplot as plt

class DataFeeder(threading.Thread):

	def __init__(self, metadata):
		super(DataFeeder, self).__init__()
		self._metadata = metadata
		self._cur = 0

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
		S = librosa.feature.melspectrogram(y=wav, sr=fs, n_fft=nsc, hop_length=nov, power=2.0, n_mels = hp.n_mels)
		S = S/(nsc**2)
		plt.imshow(S)
		plt.colorbar()
		plt.show()

		dbS = 10 * np.log10(np.maximum(S, hp.eps))

		plt.imshow(dbS)
		plt.colorbar()
		plt.show()

		return dbS

	def stft(self, wav, nsc, nov):
		S = librosa.core.stft(wav, n_fft=nsc, hop_length=nov)
		S = S/nsc
		Sxx = abs(S)

		print(S)

		print(Sxx)

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

		plt.plot(coef)
		plt.show()

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

		plt.plot(wav)
		plt.show()


		print(nsc)

		print(nov)
		specgram = self.stft(wav, nsc, nov)

		plt.imshow(specgram)
		plt.colorbar()
		plt.show()

		stft_coef = np.mean(specgram, axis=0)
		stft_coef_dB = self.to_dB(stft_coef)
		first, last = self.get_active_index(stft_coef_dB)

		plt.plot(stft_coef)
		plt.show()

		mel = self.mel_spectrogram(wav, nsc, nov, fs)
		mag = self.to_dB(specgram)

		# active_mel = mel[:, first:last]
		# active_mag = mag[:, first:last]

		# normalized_mel = active_mel/hp.max_db
		# normalized_mag = active_mag/hp.max_db



		return mel, mag, stft_coef_dB
