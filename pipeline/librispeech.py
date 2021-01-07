'''
text pipeline:
    - clean text
    - batch 
    - pad
    - one hot encode

audio pipeline:
    - clean
    - transform (if mel spectrogram) 
    - batch
    - pad
    - one hot encode (not sure)

text_audio pipeline:
    - load data
    - shuffel 
    - split into text and audio
    - apply audio and text pipeline
    - zip audio and text based on x,y

speaker pipeline:
    TBD
'''
import os, sys
# import ../
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import clean, transform # pylint: disable=imports 
import tensorflow as tf
import numpy as np
import pandas as pd 
import tensorflow_datasets as tfds







