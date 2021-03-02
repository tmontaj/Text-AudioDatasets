"""
Example for wavenet pipeline
"""
# import ../
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.load.librispeech import load  # pylint: disable=imports
from data.load.libri_what_to_download import what_to_download as wtd  # pylint: disable=imports
from data.load import safe_load  # pylint: disable=imports
import data.transform.text as t_text  # pylint: disable=imports
from hprams.test import hprams  # pylint: disable=imports
import pipeline.librispeech as pipeline  # pylint: disable=imports
import tensorflow as tf
from scipy.io.wavfile import write
import numpy as np
import data.load.librispeech as load # pylint: disable=imports


print(os.path.dirname(os.path.abspath(__file__)))

home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = os.path.join(home, "dataset")

safe_load(load, wtd, src, hprams["splits"])


x = pipeline.text_audio(src=src, split="dev-clean", **hprams["text_audio"])
# x = pipeline.text_audio(src=src, split="dev-other", **hprams["text_audio"])

x = x.take(1)
i=0

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.0001, np.max(np.abs(wav)))
    write(path, sr, wav.astype(np.int16))

# dataset = load.load_split(src=src, split="dev-clean")
# print(dataset.head())
for i in x:
    print(i)

i=0
for audio, text in x:
    audio = audio[0].numpy()
    text = text[0][2:]
    text = t_text.int2string(text)
    print("----------------")
    print(text)#.numpy())
    # print(audio)
    save_wav(wav=audio, path="test%d.wav"%(i), sr=16000)
    i+=1


# text = t_text.int2string([1,0,0,0,0])
# print(text)

#     print("__sample__")
#     print("audio", i[0])
#     print("text", i[1])
#     # print(i[1][0][0])
#     # print(i[1][0].shape)
#     print("text shape 1", tf.shape(i[1][0]))
#     print("text shape 2", tf.shape(i[1][1]))


# x = pipeline.audio_audio(src=src, split="dev-clean", 
#                          sampling_rate=16000, **hprams["audio_audio"])

# x = x.take(1)
# for i in x:
#     print(i)


# x = pipeline.speaker_verification(
#     src=src, split="dev-clean", **hprams["speaker_verification"])

# x = x.take(1)
# for i in x:
#     print(i)
