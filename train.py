# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import cPickle as pickle
import BiLSTMCRF
torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

with open("data/Traindata.pkl","rb") as inp:
    word_to_ix = pickle.load(inp)
    training_data = pickle.load(inp)

tag_to_ix = {"B": 0, "M": 1, "E": 2, "S": 3, START_TAG: 4, STOP_TAG: 5}

model = BiLSTMCRF.BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = BiLSTMCRF.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(300):  # epoch是指数据集被训练的次数，与数据集大小成反比增长
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = BiLSTMCRF.prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    print(word_to_ix)
    test_str = "爱 像 龙 卷 风".split()
    print(test_str)
    precheck_sent = BiLSTMCRF.prepare_sequence(test_str, word_to_ix)
    print(model(precheck_sent))