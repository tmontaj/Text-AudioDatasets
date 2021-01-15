import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

def audio_cleaning_old(arr, threshold):
  i=0
  print(threshold)
  while i<len(arr):
      if arr[i]>=threshold:
        arr_new= arr[i:]
        break   
      i +=1
  i=len(arr_new)-1
  while i>0:
      if arr_new[i]>=threshold:
        arr_new= arr_new[0:i+1]
        break   
      i -=1
  return arr_new


def audio_cleaning(audio, threshold):
  """
  clean audio from starting and trailing silence 
  
  Arguments:
  audio -- audio as np.array or tf tensor
  threshold -- path to the file
  Returns:
  audio -- clean audio  
  """
  bound = tfio.experimental.audio.trim(audio, axis=0, epsilon=threshold)
  start = bound[0][0]
  tail = bound[1][0]
  return audio[start:tail+1]


def tf_audio_cleaning(audio, threshold):
  return tfio.experimental.audio.trim(audio, axis=0, epsilon=threshold)