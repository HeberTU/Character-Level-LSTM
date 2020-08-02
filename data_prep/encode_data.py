
def get_vocabulary(text):
        '''Creates the vocabulary needed to represent a corpus as numerc values
            Arguments
            ---------
            text: Corpus you want represent as numerc values 

        '''
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    return chars, int2char, char2int

def get_encoded(text,char2int):
    ''' Translate characters into numeric values unign a dict
        Arguments
        ---------
        text: Corpus you want to translate into numeric values
        char2int: dictionary with characters as keys and number as values

    '''
    encoded = np.array([char2int[ch] for ch in text])
    return encoded


def one_hot_encode(arr, n_labels):
    '''Create a one hot encode from an array.
       [1,2] --> [[0,1,0],[0,0,1]]
       Arguments
       ---------
       arr: Array you want to one hot encode
       n_labels: Vocabulary size, i.e. lenght of each on hot encoded vector
    '''
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot
