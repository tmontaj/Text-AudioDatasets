import tensorflow as tf 

def string2int(text, alphabet_size = 26, first_letter=96):
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

        if text[i] == -52: # -50 = , comma
            text[i] = alphabet_size+2

    return text

def pad(texts):
    """
    padd batch os text encoded with _string2int
    Arguments:
        texts -- a batch of text to transform 
    Returns:
        texts -- a batch of padded text 
    """
    return tf.keras.preprocessing.sequence.pad_sequences(texts, maxlen=None,
                                                       dtype='int32', padding='post', value=0.0)

def one_hot_encode(texts, comma=False, alphabet_size = 26, first_letter=96):
    """
    one hot encode a batch of text padd with _padd
    Arguments:
        texts -- a batch of text to transform 
        alphabet_size -- alphabet size of the language 
        first_letter -- number of first letter in the alphabet when passed ord()
    Returns:
        texts -- a batch of one hot encodded text (letter)
    """
    depth = alphabet_size+3 if comma else alphabet_size+2
    return tf.one_hot(texts, depth=depth, axis=2, on_value=1, off_value=0) 