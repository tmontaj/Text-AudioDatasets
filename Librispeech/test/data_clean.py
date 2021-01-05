# pylint: disable=imports 
import os, sys
# import ../
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.clean.text as text
import data.clean.audio as audio

import numpy as np  


class TestClassText:
    def test_collapse_whitespace(self):
        input = "test collapse whitespaces       "
        assert text._collapse_whitespace(text=input) == "testcollapsewhitespaces"

    def test_lowercase(self):
        input = "test LOWERcAse"
        assert text._lowercase(text=input) == "test lowercase"
    
    def test_remove_commas(self):
        input = "test remove, co,mmas ,like this ,"
        assert text._remove_commas(text=input) == "test remove commas like this "
        
    def test_remove_symbols(self):
        input = r"_remove_symbols ~!@#$%^&*()_+{}:\"??><'</\|[].,"
        assert text._remove_symbols(text=input) == "removesymbols '.,"
   

    def test_expand_abbreviations(self):
        abbreviations_list = text._abbreviations_list
        input = ""
        target = ""
        for i in abbreviations_list:
            input = input + i[0] + " "
            target = target + i[1] + " "

        assert text._expand_abbreviations(text=input) == target

    def test_normalize_numbersexpand_abbreviations(self):
        pass



class TestClassAudio:

    def test_audio_cleaning_1(self):

        assert (audio.audio_cleaning(audio=np.array([0,0,00,4,5,6,6,6,6,7,8,5,5,5]),
               threshold=5) & np.array([5,6,6,6,6,7,8,5,5,5], dtype=np.int64)).any()
    
    def test_audio_cleaning_2(self):

        assert audio.audio_cleaning(audio=np.array([0,0,00,4]),
               threshold=5).size == 0

