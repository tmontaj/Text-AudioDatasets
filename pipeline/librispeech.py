'''
module overview

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
    - creat tf ds 
    - shuffel 
    - split into text and audio
    - apply audio and text pipeline
    - zip audio and text based on x,y

audio =_audio pipeline:
    - load data 
    - creat tf ds 
    - shuffel 
    - copy into audio and audio
    - apply audio and audio pipeline
    - zip audio and text based on x,y


speaker pipeline:
    TBD
'''
import os, sys
# import ../
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import clean, transform # pylint: disable=imports 

import data.clean.audio as audio # pylint: disable=imports 
import data.clean.text as text # pylint: disable=imports 

import data.transform.audio as t_audio # pylint: disable=imports 
import data.transform.text  as t_text # pylint: disable=imports 


from data import load # pylint: disable=imports 
import tensorflow as tf
import numpy as np
import pandas as pd 

def _text(dataset, batch, remove_comma, alphabet_size, first_letter):
    """
    Text pipeline  
    
    Arguments:
    dataset -- td dataset to be preprocessed 
    batch   -- batch sizee
    alphabet_size -- alphabet size of the language 
    first_letter  -- number of first letter in the alphabet when passed ord()
    remove_comma  -- flag to remove comma or not
    Returns:
    dataset -- cleaned, batched, unshufeld, tf dataset of text
    """
    dataset = dataset.map(lambda x: clean.text.clean_text(x, remove_comma))
    dataset = dataset.map(lambda x: transform.text.string2int(x, alphabet_size, first_letter))
    dataset = dataset.padded_batch(batch)
    # dataset = dataset.map(lambda x: transform.text.pad(x))
    dataset = dataset.map(lambda x: transform.text.one_hot_encode(x, remove_comma, alphabet_size, first_letter))

    return dataset


def _audio(dataset, batch, src, is_spectrogram, threshold, sampling_rate=16000):
    
    """
    Audio pipeline  
    
    Arguments:
    dataset   -- td dataset to be preprocessed 
    batch     -- batch sizee
    src       -- path to data directory
    threshold -- threshold of silence to be trimed from audio (remove silnce from start and end)
    Returns:
    dataset -- cleaned, batched, unshufeld, tf dataset of audio
    """
    
    dataset = dataset.map(lambda x: load.librispeech.load_wav(src, x)) # [0] as it return wav, sampling rate
    # dataset = dataset.map(lambda x: audio.audio_cleaning(x, threshold))
    dataset = dataset.padded_batch(batch)
    
    if is_spectrogram:
        dataset = dataset.unbatch()
        dataset = dataset.map(lambda x: transform.audio.melspectrogram(x, sampling_rate, False))
        dataset = dataset.batch(batch)

    dataset = dataset.padded_batch(batch)
    # dataset = dataset.map(lambda x: transform.audio.pad(x))
    # dataset = dataset.map(lambda x: )
    return dataset

def _split_dataset(x, idx):
    """
    used to split tf dataset into sevral datasets
    """
    return x[idx]

def text_audio(src, split, reverse, batch, threshold,
                 remove_comma=True, alphabet_size=26,
                 first_letter=96, sampling_rate=16000, buffer_size=1000):
    """
    Text and Audio pipeline  
    
    Arguments:
    src       -- path to data directory
    split     -- split name to be loaded (string) e.x. dev
    reverse   -- flag, if true dataset returned
                 is formated (audio, text) else (text, audio)
    buffer_size -- buffer size for shuffle. (Default = 1000)
    alphabet_size -- alphabet size of the language (Default = 26)
    first_letter  -- number of first letter in the alphabet when passed ord() (Default = 96)
    remove_comma  -- flag to remove comma or not (Default = True)

    Returns:
    dataset -- tf dataset of audio and text preprocessed
    """
    dataset = load.librispeech.load_split(src, split)
    dataset = dataset[["id", "text"]]
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.shuffle(buffer_size)
    
    audio_dataset = dataset.map(lambda x: _split_dataset(x=x, idx=0))
    text_dataset  = dataset.map(lambda x: _split_dataset(x=x, idx=1))

    audio_dataset = _audio(dataset=audio_dataset,
                             src=src,
                             batch=batch,
                             is_spectrogram=False,
                             threshold=threshold,
                             sampling_rate=sampling_rate)

    text_dataset  = _text(dataset=text_dataset,
                             batch=batch,
                             remove_comma=remove_comma,
                             alphabet_size=alphabet_size,
                             first_letter=first_letter)
    
    if reverse:
        dataset = tf.data.Dataset.zip((audio_dataset, text_dataset))
    else:
        dataset = tf.data.Dataset.zip((text_dataset, audio_dataset))

    return dataset


def audio_audio(src, split, reverse, batch,
                 threshold, sampling_rate=16000, buffer_size=1000):
    """
    spctrogram and audio pipeline  
    
    Arguments:
    src       -- path to data directory
    split     -- split name to be loaded (string) e.x. dev
    reverse   -- flag, if true dataset returned
                 is formated (audio, spectrogram) else (spectrogram, audio)
    batch     -- batch sizee
    threshold -- threshold of silence to be trimed from audio (remove silnce from start and end)
    sampling_rate -- audio sampling rate. Default = 16000
    buffer_size -- buffer size for shuffle. Default = 1000

    Returns:
    dataset -- tf dataset of audio and spectrogram preprocessed
    """
    dataset = load.librispeech.load_split(src, split)
    dataset["id2"] = dataset["id"]
    dataset = dataset[["id", "id2"]]
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    # dataset = dataset.shuffle(buffer_size)
    
    audio_dataset    = dataset.map(lambda x: _split_dataset(x=x, idx=0))
    spectro_dataset  = dataset.map(lambda x: _split_dataset(x=x, idx=0))

    # audio_dataset    = _audio(dataset=audio_dataset,
    #                             src=src,
    #                             batch=batch,
    #                             is_spectrogram=False,
    #                             threshold=threshold,
    #                             sampling_rate=sampling_rate)

    spectro_dataset  = _audio(dataset=spectro_dataset, 
                                src=src,
                                batch=batch,
                                is_spectrogram=True,
                                threshold=threshold,
                                sampling_rate=sampling_rate)
    
    x = spectro_dataset.take(1)
    for i in x:
        print(i)

    # if reverse:
    #     dataset = tf.data.Dataset.zip((audio_dataset, spectro_dataset))
    # else:
    #     dataset = tf.data.Dataset.zip((spectro_dataset, audio_dataset))

    return dataset