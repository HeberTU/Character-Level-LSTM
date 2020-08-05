import os
from data_prep import importdata, encode_data
from model.architecture import train_modality, CharLSTM
from model.train_model import train

text = importdata.import_corpus(path=os.getcwd()+"/Data/anna.txt")
chars, int2char, char2int = encode_data.get_vocabulary(text = text)
encoded = encode_data.get_encoded(text,char2int)



model = CharLSTM(
    chars = chars, int2char = int2char, char2int = char2int, train_on_gpu = train_modality(),
     n_hidden = 512, n_layers = 2, drop_prob = 0.25, lr = 0.001
)

print(model)

batch_size = 128
seq_length = 100
n_epochs = 20

train(
    model = model, data = encoded, train_on_gpu = train_modality(),
    epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_status=10)
