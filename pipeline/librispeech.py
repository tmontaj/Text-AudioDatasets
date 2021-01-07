'''
text pipeline:
    - clean text
    - string2num
    - batch 
    - pad
    - one hot encode

audio pipeline:
    - load wav
    - clean
    - transform (if mel spectrogram) 
    - batch
    - pad
    - one hot encode (not sure...TBD)

split data:
    - return text or audio

text_audio pipeline:
    - load data
    - creat tfds 
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
from data import load # pylint: disable=imports 
import tensorflow as tf
import numpy as np
import pandas as pd 

def text(dataset, batch, remove_comma, alphabet_size, first_letter):
    dataset = dataset.map(lambda x: clean.text.clean_text(x, remove_comma))
    dataset = dataset.map(lambda x: transform.text.string2int(x, alphabet_size, first_letter))
    dataset = dataset.batch(batch)
    dataset = dataset.map(lambda x: transform.text.pad(x))
    dataset = dataset.map(lambda x: transform.text.one_hot_encode(x, remove_comma, alphabet_size, first_letter))

    return dataset


def audio(dataset, batch, src, is_spectrogram, threshold, sampling_rate=16000):
    dataset = dataset.map(lambda x: load.librispeech.load_wav(src, x))
    dataset = dataset.map(lambda x: clean.audio.audio_cleaning(audio, threshold))
    if is_spectrogram:
        dataset = dataset.map(lambda x: transform.audio.melspectrogram(x, sampling_rate, False))
    
    dataset = dataset.batch(batch)
    dataset = dataset.map(lambda x: transform.audio.pad(x))
    # dataset = dataset.map(lambda x: )
    return dataset

def split_dataset(x, idx):
    return x[idx]

def text_audio(src, split, reverse, buffer_size=1000, **kwargs):
    dataset = load.librispeech.load_split(src, split)
    dataset = dataset[["id", "text"]]
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.shuffle(buffer_size)
    
    audio_dataset = dataset.map(lambda x: split_dataset(0, x))
    text_dataset  = dataset.map(lambda x: split_dataset(1, x))

    audio_dataset = audio(audio_dataset, **kwargs)
    text_dataset  = text(text_dataset, **kwargs)
    
    if reverse:
        dataset = tf.data.Dataset.zip((audio_dataset, text_dataset))
    else:
        dataset = tf.data.Dataset.zip((text_dataset, audio_dataset))

    return split_dataset