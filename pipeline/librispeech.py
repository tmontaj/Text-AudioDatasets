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

def load_speaker(src, speaker, id_):
    id_ = int(id_)
    id_ = speaker[id_, 1]
    sample = load.load_wav(src.decode(), id_.decode())
    sample = tf.squeeze(sample, axis =-1)
    
    return sample

def load_len_(src, speaker, id_, len_, sampling_rate, melspectrogram):
    total_len=0
    result = np.zeros((len_))
    while total_len < len_:
        sample = load_speaker(src, speaker, id_)
        if sample.shape[0]>len_:
            start = np.random.uniform(
                    size=1,
                    low=0,
                    high=sample.shape[0]-len_-1).tolist()[0]

            sample = sample[int(start):int(start)+len_]

        if sample.shape[0]+total_len>len_:
            sample = sample[0:len_-total_len]
        
        result[total_len:total_len+sample.shape[0]]=sample

        id_=(id_+1)%speaker.shape[0]
        total_len+=sample.shape[0]

    result = transform.audio.melspectrogram(
                    result, sampling_rate, False, **melspectrogram)
    return result
     


def _speaker(src, speaker, data, output_dims, num_recordes,
             threshold, sampling_rate, max_time, melspectrogram,):

    def _speaker_(src, speaker, data, output_dims, num_recordes,
                  threshold, sampling_rate, max_time):
                  
        len_ = sampling_rate*max_time
        
        recordes = np.zeros((num_recordes, output_dims[0], output_dims[1]), dtype=np.float32)
        
        speaker = data[data[:,0]==speaker]
        
        #pylint: disable=unexpected-keyword-arg
        random_sampels_idx = np.random.uniform(
            size=num_recordes,
            low=0,
            high=speaker.shape[0]-1).tolist()
        
        for i, sample in enumerate(random_sampels_idx):
            rec = load_len_(src, speaker, sample, len_, sampling_rate, melspectrogram)
            recordes[i] = rec

        return recordes

    return tf.numpy_function(_speaker_,
                             [src,
                              speaker,
                              data,
                              output_dims,
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

def set_shapes(x, y, melspectrogram):
    if melspectrogram is not {}:
        x.set_shape([None, melspectrogram["mels"]])
    y.set_shape([None])
    return (x,y)

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

    dataset = dataset.map(lambda x,y: set_shapes(x, y, melspectrogram))

    if is_spectrogram:
        dataset = dataset.padded_batch(batch, padded_shapes=([None,melspectrogram["mels"]], [None]))
    else:
        dataset = dataset.padded_batch(batch, padded_shapes=([None], [None]))
    
    if not reverse:
        dataset = dataset.map(lambda x,y: (y,x))

    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()

    return dataset

def speaker_verification(src, split, batch, melspectrogram,
                         num_recordes, threshold, max_time=5,
                         sampling_rate=16000, buffer_size=1000):

    dataset_row = load.load_split(src, split)
    
    dataset = dataset_row.speaker.unique()
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    data = dataset_row[["speaker","id"]]
    len_ = sampling_rate*max_time
    sample = np.zeros((len_), dtype=np.float32)
    output_dims = transform.audio.melspectrogram(
                    sample, sampling_rate, False, **melspectrogram).shape
    
    if buffer_size:
        dataset = dataset.shuffle(buffer_size)

    speaker_dataset = dataset.map(lambda x: _speaker(
                                                     src=src,
                                                     speaker=x,
                                                     data=data,
                                                     output_dims=output_dims,
                                                     num_recordes=num_recordes,
                                                     threshold=threshold,
                                                     melspectrogram=melspectrogram,
                                                     sampling_rate=sampling_rate,
                                                     max_time=5))

    speaker_dataset = speaker_dataset.batch(batch)
    speaker_dataset = speaker_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return speaker_dataset
