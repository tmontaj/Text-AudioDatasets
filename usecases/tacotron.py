"""
Example for tacotron pipeline
"""
# import ../
import pipeline.librispeech as pipeline  # pylint: disable=imports
# hprams
from hprams.main import hprams  # pylint: disable=imports
from data.load import safe_load  # pylint: disable=imports
from data.load.libri_what_to_download import what_to_download as wtd  # pylint: disable=imports
from data.load.librispeech import load  # pylint: disable=imports
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(os.path.dirname(os.path.abspath(__file__)))

home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = os.path.join(home, "dataset")

safe_load(load, wtd, src, hprams["splits"])


x = pipeline.text_audio(src=src, split="dev-clean", **hprams["text_audio"])
