import scipy
import numpy as np
import matplotlib.pyplot as plt
import librosa as librosa
import librosa.display
from scipy import signal
from scipy.io import wavfile


def stft(audio, sampling_rate, plot):
  """
  ALPHA don't use this function without test
  genrate stft of mono audio 

  Arguments:
  audio -- mono audio list MUST be numpy
  sampling_rate -- sampling rate of input audio 
  plot -- flag to plot genrated mel spectrogram 
  
  Return:
  Zxx -- genrated stft
  """

  f, t, Zxx = signal.stft(audio,sampling_rate)
  if plot == True :
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

  return Zxx


def plot_melsprctrogram(spectro):
  """
  plot mel spectrogram

  Arguments:
  spectro -- mel spectrogram to plot
  """
  dp=librosa.power_to_db(spectro,ref=np.max),
  librosa.display.specshow(librosa.power_to_db(dp,ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
  plt.colorbar(format='%+2.0f dB')
  plt.title('Mel spectrogram')
  plt.tight_layout()

def melspectrogram(audio, sampling_rate, plot):
  """
  genrate mel spectrogram of mono audio 

  Arguments:
  audio -- mono audio list could be either python list or numpy
  sampling_rate -- sampling rate of input audio 
  plot -- flag to plot genrated mel spectrogram 
  
  Return:
  spectro -- genrated mel spectrogram
  """
  if type(audio) == type([]):
    audio= np.array(audio, dtype=np.float32)
  else:
    audio= audio.astype(dtype=np.float32)

  spectro= librosa.feature.melspectrogram(y=audio, sr=sampling_rate, S=None, n_fft=800, hop_length=200, power=2.0 ,n_mels=80)
  
  if plot == 1 :
    plot_melsprctrogram(spectro)
  
  return spectro