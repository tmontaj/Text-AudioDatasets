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

def string2int(text, alphabet_size = 26, first_letter=97):
    """
    This is TF implementaion 
    transforn string to a list of numbers corresponding to letter index
    dot is numberd as alphabet_size + 1 and comma is alphabets_size + 2
    first letter is numberd as 1, 0 is keept for padding. Supports UTF-8 only

    Arguments:
        text -- text to transform 
        alphabet_size -- alphabet size of the language (Default : 26)
        first_letter -- number of first letter in the UTF-8 (Default : 97)

    Returns:
        text -- list of letter indexes 
    """
    text = tf.strings.unicode_decode(text, input_encoding="UTF-8")
    text = text-first_letter+1 # so first char have code = 1
    text = tf.where(text==-50, alphabet_size+1, text) # replace dot
    text = tf.where(text==-64, alphabet_size+2, text) # replace space
    text = tf.where(text==-52, alphabet_size+3, text) # replace comma

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
                                                        
def one_hot_encode(texts, comma=False, alphabet_size = 26):
    """
    one hot encode a batch of text padd with _padd
    Arguments:
        texts -- a batch of text to transform 
        alphabet_size -- alphabet size of the language (Default: 26)
        comma -- flag to detrmine if there is commas or not
    Returns:
        texts -- a batch of one hot encodded text (letter)
    """
    depth = alphabet_size+4 if comma else alphabet_size+3
    return tf.one_hot(texts, depth=depth, axis=2, on_value=1, off_value=0) 