# pylint: disable=imports 
import os, sys
# import ../
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.transform.text as text # pylint: disable=imports 
import data.transform.audio as audio # pylint: disable=imports 


import numpy as np  
import tensorflow as tf

class TestClassText:
    def test_string2int_eng(self):
        input = "abcxyz.,"
        assert text.string2int(input).numpy().tolist() == [1,2,3,24,25,26,27,28]

    def test_one_hot_encode(self):
        input = [[1 ,5, 4, 3, 0, 30]]
        output =  np.array([[
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]])
        assert np.array_equal(text.one_hot_encode(input, alphabet_size=5).numpy()
                , output)

class TestClassAudio:
    pass

