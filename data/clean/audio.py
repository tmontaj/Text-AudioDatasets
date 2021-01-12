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


@tf.function
def audio_cleaning(audio, threshold):
  """
  clean audio from starting and trailing silence 
  
  Arguments:
  audio -- audio as np.array or tf tensor
  threshold -- path to the file
  Returns:
  audio -- clean audio 
  """
  found_start = tf.constant(False, dtype=tf.bool)
  found_tail  = tf.constant(False, dtype=tf.bool)

  start = 0
  tail  = tf.shape(audio)[0]-1
  x=0

  tf.print(tail)

  run1 = tf.constant(True, dtype=tf.bool)
  run2 = tf.constant(True, dtype=tf.bool)

  while run1 and run2:
    con = (not found_start) and tf.math.greater(threshold, audio[start][0])
    tf.print("--found_start_con--")
    tf.print(tf.math.greater(audio[start][0], threshold))
    tf.print("--found_start--")
    tf.print(found_start)
    
    tf.print("--con1--")
    tf.print(con)

    found_start = tf.cond(tf.math.greater(audio[start][0], threshold), lambda: True, lambda: False)
    start = tf.cond(con, lambda: start+1, lambda: start)

    con = (not found_tail) and tf.math.greater_equal(threshold, audio[tail][0])
    tf.print("--con2--")
    tf.print(con)
    tail = tf.cond(con, lambda: tail-1, lambda: tail)
    found_tail = tf.math.greater_equal(audio[tail][0], threshold)

    run1 = tf.cond(found_start and found_tail, lambda: False, lambda: True)
    run2 = tf.cond(tf.math.greater_equal(start, tail), lambda: False, lambda: True)

    tf.print("start")
    tf.print(start)

    x=x+1
    tf.print("x")
    tf.print(x)
    
    tf.print("audio[start]")
    tf.print(audio[start][0])
    
    tf.print("tail")
    tf.print(tail)

    tf.print("--2--")
    tf.print(run1)
    tf.print("--3--")
    tf.print(run2)
    tf.print("--4--")
    tf.print(run1 and run2)

    found_start = True
    found_tail = True
  

  return audio[start:tail+1]


def tf_audio_cleaning(audio, threshold):
  return tfio.experimental.audio.trim(audio, axis=0, epsilon=threshold)