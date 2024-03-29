import scipy
import numpy as np
import matplotlib.pyplot as plt
import librosa as librosa
import librosa.display
from scipy import signal
from scipy.io import wavfile
import tensorflow_io as tfio
import tensorflow as tf


def librosa_stft(audio, sampling_rate, plot):
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


def librosa_plot_melsprctrogram(spectro):
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

def librosa_melspectrogram(audio, sampling_rate, plot):
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



def plot_melsprctrogram(spectro):
  """
  plot mel spectrogram

  Arguments:
  spectro -- mel spectrogram to plot

  """

  dp = tfio.experimental.audio.dbscale(spectro, top_db=80)
  
  plt.figure()
  plt.imshow(dp.numpy())

def melspectrogram(audio, sampling_rate, plot, nfft=800, window=512,
                   stride=200, mels=80, fmin=0, fmax=8000):
  """
  genrate mel spectrogram of mono audio 

  Arguments:
  audio -- mono audio list could be either python list or tf (NOT numpy)
  sampling_rate -- sampling rate of input audio 
  plot -- flag to plot genrated mel spectrogram 
  
  Return:
  spectro -- genrated mel spectrogram
  """
  if type(audio) == type([]):
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)
  else:
    audio = tf.cast(audio, dtype=tf.float32) # pylint: disable=[unexpected-keyword-arg, no-value-for-parameter]

  spectro = tfio.experimental.audio.spectrogram(audio, nfft=nfft, window=window, stride=stride)
  spectro = tfio.experimental.audio.melscale(spectro, rate=sampling_rate, mels=mels, fmin=fmin, fmax=fmax)
  
  if plot == True:
    plot_melsprctrogram(spectro)
  
  return spectro

def pad(audios):
  tf.print(audios)
  
  def _pad(audios):
      """
      padd batch os mono audio (Not working with tf batch)
      Arguments:
          audios -- a batch of audio to transform 
      Returns:
          audios -- a batch of padded audio 
      """
      return tf.keras.preprocessing.sequence.pad_sequences(audios, maxlen=None,
                                                        dtype='int32', padding='post', value=0.0)
  return tf.numpy_function(_pad, [audios], [tf.float32])

def inverse_melspectrogram(spectro, sampling_rate, length, nfft=800, window=512,
                   stride=200, mels=80, fmin=0, fmax=8000):
  """
  genrate mel spectrogram of mono audio 

  Arguments:
  spectro -- mel spectrogram
  sampling_rate -- sampling rate of input audio 
  length -- len_ of input without padding
  For the rest of the parametars please read TF docuentation
  
  Return:
  audio -- mono audio list
  """
  
  spectro = spectro[:length, :]
  spectro = tf.stack(spectro, axis=1)
  spectro = spectro.numpy()

  audio = librosa.feature.inverse.mel_to_audio(spectro, sr=sampling_rate, fmin=fmin, fmax=fmax,
                                                        n_fft=nfft, win_length=window, hop_length=stride)
  return audio
