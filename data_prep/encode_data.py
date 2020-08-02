
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


def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size * seq_length from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''

    ## Get the number of batches we can make
    n_batches = int(len(arr)/(batch_size*seq_length))

    ## Keep only enough characters to make full batches
    arr = arr[:batch_size*seq_length *n_batches]

    ## Reshape into batch_size rows
    arr = arr.reshape((batch_size,-1))

    ## Iterate over the batches using a window of size seq_length
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1] = x[:, 1:]
            y[:,-1] = arr[:,n+seq_length]
        except IndexError:
            y[:, :-1] = x[:, 1:]
            y[:,-1] = arr[:,0]
        yield x, y
