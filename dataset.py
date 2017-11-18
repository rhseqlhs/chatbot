import math
import random
import numpy as np
import torch
import torchtext.vocab as vocab
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from data_utils import process_data, convert_to_index, prune_data


class ChatDataset(Dataset):
    """
    Twitter chat dataset.
    """

    def __init__(self, data_path, max_length, max_vocab_size, min_freq,
                 eos_token, pad_token, unk_token, embed_dim, special_tokens,
                 threshold, pre_trained=False):
        """
        Args:
            data_path (str): path to data file
            max_length (int): maximum length of each sentence, including <eos>
            max_vocab_size (int): maximum number of words allowed in vocabulary
            min_freq (int): minimum frequency to add word to vocabulary
            eos_token (str): end of sentence token (tells decoder to start or stop)
            pad_token (str): padding token
            unk_token (str): unknown word token
            embed_dim (int): dimension of embedding vectors
            special_tokens (list of str): other tokens to add to vocabulary
            threshold (int): count of unknown words required to prune sentence
            pre_trained (Vector): pre trained word embeddings
        """
        special_tokens = [pad_token, unk_token, eos_token] + special_tokens
        # the value 0 will be regarded as padding
        assert special_tokens[0] == pad_token
        inputs, targets, counter, xlen = process_data(data_path, max_length,
                                                      eos_token, pad_token)
        self.vocab = vocab.Vocab(counter=counter, max_size=max_vocab_size,
                                 min_freq=min_freq, specials=special_tokens)
        if pre_trained is not False:
            self.vocab.load_vectors(pre_trained)
        assert len(inputs) == len(targets) and len(inputs) == len(xlen)

        self.nwords = len(self.vocab)
        self.max_len = max_length
        self.eos_idx = self.vocab.stoi[eos_token]
        self.pad_idx = self.vocab.stoi[pad_token]
        self.unk_idx = self.vocab.stoi[unk_token]
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.embed_dim = embed_dim
        self.unk_count = 0  # number of unknown words in dataset
        self.total_tokens = 0  # number of tokens in dataset not counting padding
        self.special_tokens = special_tokens
        self.x_lens = xlen
        self.x_data = np.zeros((len(inputs), max_length), dtype=np.int32)
        self.y_data = np.zeros((len(targets), max_length), dtype=np.int32)

        convert_to_index(inputs, self, self.x_data)
        convert_to_index(targets, self, self.y_data)
        self.x_data, self.y_data, self.x_lens = prune_data(self.x_data, self.y_data,
                                                           self.x_lens, self, threshold)
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)

    def __len__(self):
        return len(self.x_lens)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx], self.x_lens[idx]

    def eos_tensor(self, size, use_cuda):
        """
        Return tensor representing end of sentence token.
        """
        eos = torch.LongTensor(size).fill_(self.eos_idx)
        if use_cuda:
            eos = eos.cuda()
        return eos


def collate_fn(data):
    """
    Creates a mini-batch, overriding default_collate function
    in order to provide batches with input sorted by length.
    """
    # sort such that input line lengths are in decreasing order
    # requirement for using torch.nn.utils.rnn.pack_packed_sequence
    data.sort(key=lambda x: x[-1], reverse=True)

    # group input sentences, target sentences, and lengths together
    inputs, targets, lens = zip(*data)
    assert len(inputs) == len(targets)

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    return inputs, targets, list(lens)


def split_data(dataset, train, valid, test):
    """
    Split data to training, validation, and test set by returning
    samplers used to load batches from train, valid, and test sets.

    Args:
        dataset (ChatDataset): dataset from which to split
        train (float): training set proportion
        valid (float): validation set proportion
        test (float): test set proportion

    Precondition: train + valid + test = 1 and none are zero
    """
    if train > 1 or valid > 1 or test > 1 or train < 0 or valid < 0 or test < 0:
        raise ValueError("Please provide valid split proportions.")
    elif train + valid + test != 1:
        raise ValueError("Please make sure proportions add to one.")
    elif train == 0 or valid == 0 or test == 0:
        raise ValueError("All of the split proportions must be non zero.")

    train_end_idx = math.ceil(len(dataset) * train)
    test_start_idx = math.floor(len(dataset) * test)

    # SubsetRandomSampler takes in lists, so shuffle a list
    sentence_indexes = list(range(len(dataset)))
    random.shuffle(sentence_indexes)

    train_list = sentence_indexes[:train_end_idx]
    valid_list = sentence_indexes[train_end_idx:-test_start_idx]
    test_list = sentence_indexes[-test_start_idx:]

    # Samplers for loading batches
    train_sampler = SubsetRandomSampler(train_list)
    valid_sampler = SubsetRandomSampler(valid_list)
    test_sampler = SubsetRandomSampler(test_list)

    return train_sampler, valid_sampler, test_sampler
