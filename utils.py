# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display as librosadisplay

from IPython.display import Audio
import IPython

def plot_stft(x, sample_rate, show_black_and_white=False):
  stft = librosa.stft(x, n_fft=2048)
  x_stft = np.abs(stft)
  fig, ax = plt.subplots()
  fig.set_size_inches(20, 10)
  x_stft_db = librosa.amplitude_to_db(x_stft, ref=np.max)
  if(show_black_and_white):
    librosadisplay.specshow(data=x_stft_db, y_axis='log', 
                             sr=sample_rate, cmap='gray_r')
  else:
    librosadisplay.specshow(data=x_stft_db, y_axis='log', sr=sample_rate)

  plt.colorbar(format='%+2.0f dB')
  plt.show()


def play_audio(audio_samples, sample_rate):
    IPython.display.display(Audio(audio_samples, rate=sample_rate))