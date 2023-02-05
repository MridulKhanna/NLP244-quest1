import os
from io import open
import torch


class Vocabulary(object):
    def __init__(self):
        self.type2index = {}
        self.idx2type = []

    def add_type(self, token):
        if token not in self.type2index:
            self.idx2type.append(token)
            self.type2index[token] = len(self.idx2type) - 1
        return self.type2index[token]

    def __len__(self):
        return len(self.idx2type)

unk_count = 0
class Corpus(object):
    def __init__(self, path):
        self.vocab = Vocabulary()
        self.train = self.tokenize(os.path.join(path, "wiki.train.tokens"))
        self.valid = self.tokenize(os.path.join(path, "wiki.valid.tokens"))
        self.test = self.tokenize(os.path.join(path, "wiki.test.tokens"))

    def tokenize(self, path):
        """Tokenizes a text file."""
        global unk_count
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    self.vocab.add_type(word)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ["<eos>"]
                ids = []
                for word in words:
                    if "wiki.train.tokens" in path and word =="<unk>":
                        unk_count = unk_count + 1
                    ids.append(self.vocab.type2index[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

corpusobj = Corpus("D:\\NLP244-quest1-main\\NLP244-quest1-main\\data\\wikitext-2")
temp = vars(corpusobj)['train']
print(unk_count)
print("Percentage of <unk> tokens in training dataset:", (unk_count/len(temp))*100)