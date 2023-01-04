import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
import re
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer



class DataLoader:

    def __init__(self, tokenize):
        self.seed = 42
        self.tokenize = tokenize
        print('dataset initializing start')

    def make_dataset(self, data, train_size=0.8):
        data.iloc[:, 0] = data.iloc[:, 0].apply(lambda x : x.replace("<br />", " "))
        data.iloc[:, 0] = data.iloc[:, 0].apply(lambda x : x.lower())
        
        data["len"] = data.iloc[:, 0].apply(lambda x : len(x.split()))
        data = data[data["len"] < 256]
        print("Tokenizing the data...")
        data["len"] = data.iloc[:, 0].apply(lambda x : len(self.tokenize(x)))
        data = data[data["len"] < 256]
        
        print("Length of the data : ", len(data))

        train, rem = train_test_split(data, train_size=train_size, random_state = self.seed)
        valid_size = 0.5
        valid, test = train_test_split(rem, train_size=valid_size, random_state = self.seed)
        print(train.iloc[:, 1].iloc[0])

        train.iloc[:, 0] = train.iloc[:, 0].apply(lambda row: re.sub("[^A-Za-z]+", " ", row)).apply(self.tokenize)
        valid.iloc[:, 0] = valid.iloc[:, 0].apply(lambda row: re.sub("[^A-Za-z]+", " ", row)).apply(self.tokenize)
        test.iloc[:, 0] = test.iloc[:, 0].apply(lambda row: re.sub("[^A-Za-z]+", " ", row)).apply(self.tokenize)

        return train, valid, test

    def get_vocab(self, training_corpus):
        vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2}
        for item in training_corpus:
            for word in item:
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab

    def make_iter(self, train, validate, test, batch_size, device):
        vocab = self.get_vocab(train.iloc[:, 0])  # TODO: I added this line - otherwise vocab is not defined - correct?

        print(train.iloc[0])
        train_y = torch.tensor(train.iloc[:, 1].values.astype(np.float32), device=device)
        valid_y = torch.tensor(validate.iloc[:, 1].values.astype(np.float32), device=device)
        test_y = torch.tensor(test.iloc[:, 1].values.astype(np.float32), device=device)

        unk_ID = vocab["__UNK__"]

        train_x_tensor = []
        for idx, text_corpus in enumerate(tqdm(train.iloc[:, 0])):
            foo = []
            for token in text_corpus:
                word_ID = vocab.get(token, unk_ID)
                foo.append(word_ID)
            while len(foo) < 256:
                foo.append(vocab["__PAD__"])
            train_x_tensor.append(foo)

        valid_x_tensor = []
        for idx, text_corpus in enumerate(tqdm(validate.iloc[:, 0])):
            foo = []
            for token in text_corpus:
                word_ID = vocab.get(token, unk_ID)
                foo.append(word_ID)
            while len(foo) < 256:
                foo.append(vocab["__PAD__"])
            valid_x_tensor.append(foo)

        test_x_tensor = []
        for idx, text_corpus in enumerate(tqdm(test.iloc[:, 0])):
            foo = []
            for token in text_corpus:
                word_ID = vocab.get(token, unk_ID)
                foo.append(word_ID)
            while len(foo) < 256:
                foo.append(vocab["__PAD__"])
            test_x_tensor.append(foo)

        train_x = torch.tensor(train_x_tensor, device=device)
        valid_x = torch.tensor(valid_x_tensor, device=device)
        test_x = torch.tensor(test_x_tensor, device=device)

        train = torch.utils.data.TensorDataset(train_x, train_y)
        validate = torch.utils.data.TensorDataset(valid_x, valid_y)
        test = torch.utils.data.TensorDataset(test_x, test_y)

        train_iterator = torch.utils.data.DataLoader(dataset = train, batch_size = batch_size, shuffle = True)
        valid_iterator = torch.utils.data.DataLoader(dataset = validate, batch_size = batch_size, shuffle = True)
        test_iterator = torch.utils.data.DataLoader(dataset = test, batch_size = batch_size, shuffle = True)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator

class Tokenizer:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def tokenize(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))