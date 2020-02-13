#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from PyEMD import EMD, EEMD

#setting
data_input_filename = 'timeseries_tj_selor'

def hann(total_data):
	hann_array = np.zeros(total_data)
	for i in range(total_data):
		hann_array[i] = 0.5 - 0.5 * np.cos((2 * np.pi * i) / (total_data - 1))
	return hann_array

def hamm(total_data):
	hann_array = np.zeros(total_data)
	for i in range(total_data):
		hann_array[i] = 0.5386 - 0.46164 * np.cos((2 * np.pi * i) / (total_data - 1))
	return hann_array

data_input = open(data_input_filename).read().split('\n')

if data_input[-1] == '':
	data_input = data_input[:-1]
data_input = np.asarray(data_input).astype('float32')

print(data_input)
eemd = EEMD()
EIMFs = eemd(data_input)

print(len(EIMFs))

#plot
#plt.subplot(3, 1, 1)
#plt.plot(np.arange(timeseries_param_len), IMFs[0], '-', lw=2)
#plt.subplot(3, 1, 2)
#plt.plot(np.arange(timeseries_param_len), IMFs[1], '-', lw=2)
#plt.subplot(3, 1, 3)
#plt.plot(np.arange(timeseries_param_len), IMFs[2], '-', lw=2)
#plt.tight_layout()
#plt.show()

print(np.shape(data_input), np.shape(EIMFs))

#plot
#for i in range(len(EIMFs)):
#	plt.subplot(len(EIMFs), 1, i+1)
#	plt.plot(np.arange(len(data_input)), EIMFs[i], '-', lw=2)
#plt.show()

#rata-rata hujan
rata_hujan = np.reshape(data_input, (12, -1))
rata_hujan = np.mean(rata_hujan, axis=1)
print(np.shape(rata_hujan))

plt.ylabel("Curah Hujan")
plt.xlabel("Bulan")
plt.plot(np.arange(12)+1, rata_hujan)
plt.show()

#FFT

#normalize
data_input = (data_input - np.mean(data_input)) / np.std(data_input)
time = range(len(data_input))
fftdata = np.fft.fft(data_input)
fftdatafreq = np.zeros((len(data_input)))

for i in range(len(fftdata)):
	fftdatafreq[i] = abs(fftdata[i].real)

#plt.ylabel("Amplitude")
#plt.xlabel("Frequency")
#plt.plot(time[:len(fftdatafreq) // 2], fftdatafreq[:len(fftdatafreq) // 2])

#plt.show()

