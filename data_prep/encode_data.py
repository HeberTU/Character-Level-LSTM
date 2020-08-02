
def get_vocabulary(text):
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    return chars, int2char, char2int

def get_encoded(text,char2int):
    encoded = np.array([char2int[ch] for ch in text])
    return encoded
