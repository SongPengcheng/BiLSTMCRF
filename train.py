# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import BiLSTMCRF
torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
#准备用于模型训练的数据，并且，将数据处理成(字符:序号)的形式
'''
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]
'''
training_data = [(
    "我 爱 北 京 ， 天 安 门 是 故 宫 的 大 门".split(),
    "O O B I O B I I O B I O B I".split()
), (
    "溥 仪 是 中 国 最 后 一 个 皇 帝".split(),
    "B I O B I B I O O B I".split()
)]
word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)#将训练数据中的单词映射为在句子中的序号

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTMCRF.BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = BiLSTMCRF.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = BiLSTMCRF.prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

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
    test_str = "溥 仪 爱 故 宫 的 大 门 天 安 门".split()
    print(test_str)
    precheck_sent = BiLSTMCRF.prepare_sequence(test_str, word_to_ix)
    print(model(precheck_sent))