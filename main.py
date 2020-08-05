import os
from data_prep import importdata, encode_data
from model.architecture import train_modality, CharLSTM

text = importdata.import_corpus(path=os.getcwd()+"/Data/anna.txt")
chars, int2char, char2int = encode_data.get_vocabulary(text = text)
encoded = encode_data.get_encoded(text,char2int)



model = CharLSTM(
    chars = chars, int2char = int2char, char2int = char2int,train_on_gpu = train_modality(),
     n_hidden = 256, n_layers = 2, drop_prob = 0.25, lr = 0.001
)

print(model)
