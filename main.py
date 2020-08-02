import os
from data_prep import importdata, encode_data

text = importdata.import_corpus(path=os.getcwd()+"/Data/anna.txt")
chars, int2char, char2int = encode_data.get_vocabulary(text = text)
