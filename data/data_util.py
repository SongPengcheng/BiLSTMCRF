import pickle
sourcefile = open("origin.txt","r",encoding="UTF-8")
word_to_ix = {}
training_data = list()
lines = sourcefile.readlines()
for item in lines:
    wordlist = list()
    taglist = list()
    sentence = item.split()
    for train_data in sentence:
        word = train_data.split('/')[0]
        tag = train_data.split('/')[1]
        wordlist.append(word)
        taglist.append(tag)
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    training_data.append((wordlist, taglist))
with open("Traindata.pkl","wb") as target:
    pickle.dump(word_to_ix, target)
    pickle.dump(training_data,target)