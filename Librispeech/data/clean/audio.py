import numpy as np

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
  found_start = False
  found_tail  = False

  start = 0
  tail  = audio.shape[0]-1

  while True:
    if found_start and found_tail: break
    if start >= tail: break
    
    if not found_start and audio[start]<threshold:
      start+=1
    else: 
      found_start = True

    if not found_tail and audio[tail]<threshold:
      tail-=1
    else: 
      found_tail = True

  return audio[start:tail+1]
