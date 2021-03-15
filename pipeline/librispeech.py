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
from scipy.io.wavfile import write
def save_wav(wav, path, sr):
    wav *= 32767 / max(0.0001, np.max(np.abs(wav)))
    write(path, sr, wav.astype(np.int16)) 

def printer(x):
    tf.print("||||||printer|||||||")
    tf.print(x.shape)
    return x


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
    dataset = dataset.sort_values(by=['id'], ascending=True)
    dataset = tf.data.Dataset.from_tensor_slices((dataset["id"], dataset["text"]))

    if buffer_size:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(lambda x,y: (load.load_wav(src, x), y))
                                # if thid line is uncommented the x and y gets shuffled
                                # num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x,y: (audio.audio_cleaning(x, threshold),y),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x,y: (tf.squeeze(x, axis=-1),y),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_spectrogram:
        dataset = dataset.map(lambda x,y: (transform.audio.melspectrogram(
            x, sampling_rate, False, **melspectrogram), y),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
    dataset = dataset.map(lambda x,y: (x,transform.text.string2int(
        y, alphabet_size, first_letter, len_)))#,num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if one_hot_encode:
        dataset = dataset.map(lambda x,y: (x,transform.text.one_hot_encode(
            y, remove_comma, alphabet_size, len_)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.map(lambda x,y: (x, tf.concat((tf.shape(x), y), axis=0)))

    if is_spectrogram:
        dataset = dataset.padded_batch(batch, padded_shapes=([None,None], [None]))
    else:
        dataset = dataset.padded_batch(batch, padded_shapes=([None], [None]))
    
    if not reverse:
        dataset = dataset.map(lambda x,y: (y,x))

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
