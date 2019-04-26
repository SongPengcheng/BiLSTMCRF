import torch
import BiLSTMCRF
import cPickle as pickle

with open("data/Traindata.pkl","rb") as inp:
    word_to_ix = pickle.load(inp)
model = torch.load("./model/model.pkl")
test_str = "谁 的 代 表 作 是 龙 卷 风".split()
precheck_sent = BiLSTMCRF.prepare_sequence(test_str, word_to_ix)
print(model(precheck_sent))