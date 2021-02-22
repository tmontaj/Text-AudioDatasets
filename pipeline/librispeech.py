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
import os
import sys
# import ../
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
from data import clean, transform  # pylint: disable=imports
import data.clean.audio as audio  # pylint: disable=imports
import data.clean.text as text  # pylint: disable=imports
import data.transform.audio as t_audio  # pylint: disable=imports
import data.transform.text as t_text  # pylint: disable=imports
import data.load.librispeech as load # pylint: disable=imports
import pandas as pd
 

def printer(x):
    tf.print("||||||printer|||||||")
    tf.print(x.shape)
    return x


def _text(dataset, batch, remove_comma, alphabet_size, first_letter,
           len_=False, one_hot_encode=False):
    """
    Text pipeline  

    Arguments:
    dataset -- td dataset to be preprocessed (text)
    batch   -- batch sizee
    alphabet_size -- alphabet size of the language 
    first_letter  -- number of first letter in the alphabet when passed ord()
    remove_comma  -- flag to remove comma or not
    len_  -- flag to return length of text witout padding
    dataset -- td dataset of text length (len(text))

    Returns:
    dataset -- cleaned, batched, unshufeld, tf dataset of text
                (if len_=True then return (text, len(text)))
    """
    dataset = dataset.map(lambda x: transform.text.string2int(
        x, alphabet_size, first_letter, len_),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(batch)

    if one_hot_encode:
        dataset = dataset.map(lambda x: transform.text.one_hot_encode(
            x, remove_comma, alphabet_size, len_),num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


def _audio(dataset, batch, src, is_spectrogram,
           threshold, melspectrogram={}, sampling_rate=16000):
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

    dataset = dataset.map(lambda x: load.load_wav(src, x),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: audio.audio_cleaning(x, threshold),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: tf.squeeze(x, axis =-1),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset_len = dataset.map(lambda x: tf.shape(x),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_spectrogram:
        dataset = dataset.map(lambda x: transform.audio.melspectrogram(
            x, sampling_rate, False, **melspectrogram),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.padded_batch(batch)

    # dataset = dataset.unbatch()

    return dataset, dataset_len


def _speaker(speaker, data, num_recordes,
             threshold, melspectrogram,
             sampling_rate, max_time=5):

    def _speaker_(speaker, data, num_recordes,
                  threshold, sampling_rate, max_time=5):

        src, split = data[0].decode(), data[1].decode()
        speaker = speaker.decode()

        data = load.load_split(src, split)

        recordes = []
        speaker = data[data["speaker"] == speaker]
        len_ = sampling_rate*max_time
        speaker = speaker[speaker["audio_len"]>=len_]
        
        #pylint: disable=unexpected-keyword-arg
        rand_idx = np.random.uniform(
            size=num_recordes,
            low=0,
            high=speaker.shape[0]).tolist()
        
        for idx in rand_idx:
            idx = int(idx)
            rec = speaker.iloc[idx, :]
            id_ = rec["id"]

            sample = load.load_wav(src, id_)
            sample = tf.squeeze(sample, axis =-1)
            
            start = np.random.uniform(
                    size=1,
                    low=0,
                    high=sample.shape[0]-len_-1).tolist()[0]

            sample = sample[int(start):int(start)+len_]
            sample = transform.audio.melspectrogram(
                    sample, sampling_rate, False, **melspectrogram)

            recordes.append(sample)

        recordes = tf.stack(recordes)
        return recordes

    return tf.numpy_function(_speaker_,
                             [speaker,
                              data,
                              num_recordes,
                              threshold,
                              sampling_rate,
                              max_time],
                             tf.float32)

def _split_dataset(x, idx):
    """
    used to split tf dataset into sevral datasets
    """
    return x[idx]


def text_audio(src, split, reverse, batch, threshold,
               is_spectrogram, melspectrogram={}, remove_comma=True,
               alphabet_size=26, first_letter=96, sampling_rate=16000,
               buffer_size=1000, len_=False, one_hot_encode=False):
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
    one_hot_encode -- flag to return text as one hot encode or as index of it (Default = False, index)
    Returns:
    dataset -- tf dataset of audio and text preprocessed
    """
    dataset = load.load_split(src, split)

    dataset["text"] = dataset["text"].map(
        lambda x: text.clean_text(x, remove_comma))
    dataset = dataset[["id", "text"]]
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    if buffer_size:
        dataset = dataset.shuffle(buffer_size)

    audio_dataset = dataset.map(lambda x: _split_dataset(x=x, idx=0),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    text_dataset = dataset.map(lambda x: _split_dataset(x=x, idx=1),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    audio_dataset, audio_dataset_len = _audio(dataset=audio_dataset,
                           src=src,
                           batch=batch,
                           is_spectrogram=is_spectrogram,
                           melspectrogram=melspectrogram,
                           threshold=threshold,
                           sampling_rate=sampling_rate)

    text_dataset = _text(dataset=text_dataset,
                         batch=batch,
                         remove_comma=remove_comma,
                         alphabet_size=alphabet_size,
                         first_letter=first_letter,
                         len_=len_,
                         one_hot_encode=one_hot_encode)
    text_dataset = text_dataset.unbatch()
    text_dataset = tf.data.Dataset.zip((audio_dataset_len, text_dataset))
    text_dataset = text_dataset.map(lambda len_, text: tf.concat((len_, text), axis=0)) 
    text_dataset = text_dataset.batch(batch)

    if reverse:
        dataset = tf.data.Dataset.zip((audio_dataset, text_dataset))
    else:
        dataset = tf.data.Dataset.zip((text_dataset, audio_dataset))
    
    # dataset = dataset.batch(batch)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def audio_audio(src, split, reverse, batch, melspectrogram,
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
    dataset = load.load_split(src, split)
    dataset["id2"] = dataset["id"]
    dataset = dataset[["id", "id2"]]
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    if buffer_size:
        dataset = dataset.shuffle(buffer_size)

    audio_dataset = dataset.map(lambda x: _split_dataset(x=x, idx=0),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    spectro_dataset = dataset.map(lambda x: _split_dataset(x=x, idx=0),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

    audio_dataset = _audio(dataset=audio_dataset,
                           src=src,
                           batch=batch,
                           is_spectrogram=False,
                           melspectrogram=melspectrogram,
                           threshold=threshold,
                           sampling_rate=sampling_rate)

    spectro_dataset = _audio(dataset=spectro_dataset,
                             src=src,
                             batch=batch,
                             is_spectrogram=True,
                             melspectrogram=melspectrogram,
                             threshold=threshold,
                             sampling_rate=sampling_rate)

    if reverse:
        dataset = tf.data.Dataset.zip((audio_dataset, spectro_dataset))
    else:
        dataset = tf.data.Dataset.zip((spectro_dataset, audio_dataset))
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def speaker_verification(src, split, batch, melspectrogram,
                         num_recordes, threshold, max_time=5,
                         sampling_rate=16000, buffer_size=1000):

    dataset = load.load_split(src, split)
    len_ = sampling_rate*max_time
    dataset = dataset[dataset["audio_len"]>=len_]
    dataset = dataset["speaker"]
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    if buffer_size:
        dataset = dataset.shuffle(buffer_size)

    speaker_dataset = dataset.map(lambda x: _speaker(speaker=x,
                                                     data=(src, split),
                                                     num_recordes=num_recordes,
                                                     threshold=threshold,
                                                     melspectrogram=melspectrogram,
                                                     sampling_rate=sampling_rate,
                                                     max_time=5))#,
                                                     #num_parallel_calls=tf.data.experimental.AUTOTUNE)

    speaker_dataset = speaker_dataset.batch(batch)
    # speaker_dataset = speaker_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return speaker_dataset
