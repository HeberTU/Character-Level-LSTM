import os
from data_prep import importdata

text = importdata.import_corpus(path=os.getcwd()+"/Data/anna.txt")
