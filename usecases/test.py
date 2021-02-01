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
from hprams.test import hprams  # pylint: disable=imports
import pipeline.librispeech as pipeline  # pylint: disable=imports



print(os.path.dirname(os.path.abspath(__file__)))

home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = os.path.join(home, "dataset")

safe_load(load, wtd, src, hprams["splits"])


x = pipeline.text_audio(src=src, split="dev-clean", **hprams["text_audio"])

x = x.take(1)
for i in x:
    print(i)


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
