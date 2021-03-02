# pay;int: disable=no-value-for-parameter

import tensorflow as tf 

def non_graph_string2int(text, alphabet_size = 26, first_letter=96):
    """
    transforn string to a list of numbers corresponding to letter index
    dot is numberd as alphabet_size + 1 and comma is alphabets_size + 2
    first letter is numberd as 1, 0 is keept for paddind 

    Arguments:
        text -- text to transform 
        alphabet_size -- alphabet size of the language 
        first_letter -- number of first letter in the alphabet when passed ord()

    Returns:
        text -- list of letter indexes 
    """
    text = [ord(letter)-first_letter for letter in text ]
    for i in range(len(text)):
        if text[i] == -50: # -50 = . dot
            text[i] = alphabet_size+1

        if text[i] == -52: # -52 = , comma
            text[i] = alphabet_size+2

    return text

def string2int(text, alphabet_size = 26, first_letter=97, len_=True):
    """
    This is TF implementaion 
    transforn string to a list of numbers corresponding to letter index
    dot is numberd as alphabet_size + 1 and comma is alphabets_size + 2
    first letter is numberd as 1, 0 is keept for padding. Supports UTF-8 only

    Arguments:
        text -- text to transform 
        alphabet_size -- alphabet size of the language (Default : 26)
        first_letter -- number of first letter in the UTF-8 (Default : 97)
        len_  -- flag to return length of text as first element

    Returns:
        text -- list of letter indexes 
    """
    text = tf.strings.unicode_decode(text, input_encoding="UTF-8")
    text = text-first_letter+1
    text = tf.where(text==(46-first_letter+1), alphabet_size+1, text) # replace dot
    text = tf.where(text==(32-first_letter+1), alphabet_size+2, text) # replace space
    text = tf.where(text==(44-first_letter+1), alphabet_size+3, text) # replace comma
    
    text = tf.concat(values=(tf.shape(text), text), axis=0)
    
    return text

def pad(texts):
    def _pad(texts):
        """
        padd batch os text encoded with _string2int (Not working with tf batch)
        Arguments:
            texts -- a batch of text to transform 
        Returns:
            texts -- a batch of padded text 
        """
        return tf.keras.preprocessing.sequence.pad_sequences(texts, maxlen=None,
                                                              dtype='int32', padding='post', value=0.0)
            
    
    return tf.numpy_function(_pad, [texts], tf.string)
                                                        
def one_hot_encode(texts, comma=False, alphabet_size = 26, len_=True):
    """
    one hot encode a batch of text padd with _padd
    Arguments:
        texts -- a batch of text to transform 
        alphabet_size -- alphabet size of the language (Default: 26)
        comma -- flag to detrmine if there is commas or not
        len_  -- flag to return length of text as first vector
    Returns:
        texts -- a batch of one hot encodded text (letter)
    """
    depth = alphabet_size+4 if comma else alphabet_size+3
    #texts.shape = [batch, time]

    if len_:
        lengths = texts[:,0]
        lengths = tf.expand_dims(lengths, -1)
        texts = texts[:,1:]
        start = tf.repeat(lengths, repeats=depth, axis=-1)
        start = tf.expand_dims(start, axis=1)

    text_only = tf.one_hot(texts, depth=depth, axis=2, on_value=1, off_value=0) 

    if len_:
        return tf.concat((start, text_only), axis=1)
    
    return  text_only

def one_hot_decode(one_hot):
    """
    decode a one hot encode into numbers
    Arguments:
        one_hot -- one_hot encode
    Returns:
        index -- list of ints
    """
    index = tf.argmax(one_hot, axis=0)

    return index

def int2string(ints, alphabet_size = 26, first_letter=97, len_=True):
    """
    This is TF implementaion 
    transforn string to a list of numbers corresponding to letter index
    dot is numberd as alphabet_size + 1 and comma is alphabets_size + 2
    first letter is numberd as 1, 0 is keept for padding. Supports UTF-8 only

    Arguments:
        ints -- list of int to transform
        alphabet_size -- alphabet size of the language (Default : 26)
        first_letter -- number of first letter in the UTF-8 (Default : 97)
        len_  -- flag to return length of text as first element

    Returns:
        text -- list of letter indexes 
    """
    ints = ints[ints != 0]
    ints = ints+first_letter-1

    ints = tf.where(ints==(first_letter-1)+alphabet_size+1, 46, ints) # replace dot
    ints = tf.where(ints==(first_letter-1)+alphabet_size+2, 32, ints) # replace space
    ints = tf.where(ints==(first_letter-1)+alphabet_size+3, 44, ints) # replace comma

    ints = tf.strings.unicode_encode(ints, output_encoding="UTF-8")
    # ints = ints.numpy().decode('UTF-8')
    return ints
