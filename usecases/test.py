# import ../
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# data load imports
from data.load.librispeech import load # pylint: disable=imports
from data.load.libri_what_to_download import what_to_download as wtd # pylint: disable=imports
from data.load import safe_load # pylint: disable=imports

# hprams
from hprams.test import hprams  # pylint: disable=imports 

#  pipeline imports
import pipeline.librispeech as pipeline # pylint: disable=imports 

# hprams.splits

print(os.path.dirname(os.path.abspath(__file__)))

home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = os.path.join(home, "dataset")

safe_load(load, wtd, src, ["dev-clean"])

# pipeline.audio_audio(src=src, batch=3, split="dev-clean", reverse=False, threshold=5)

pipeline.text_audio(src=src, split="dev-clean", reverse="False", batch=2, threshold=5,
                 is_spectrogram=False, remove_comma=True, alphabet_size=26, 
                 first_letter=96, sampling_rate=16000, buffer_size=1000)