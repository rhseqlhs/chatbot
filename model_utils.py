import torch
import torch.nn as nn
import numpy as np
from random import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_utils import process_sentence, sentence_to_index


def train(encoder, decoder, batch_size, batches, encoder_opt, decoder_opt,
          dataset, choices, loss):
    # turn on train mode to activate dropout between layers
    encoder.train()
    decoder.train()
    use_cuda = encoder.is_cuda()
    encoder_hidden = encoder.init_hidden(batch_size)
    decoder_hidden = decoder.init_hidden(batch_size)

    total_loss = 0
    for lines, target, xlens in batches:
        sentence_len = target.size()[1]
        batch_loss = 0

        lines = Variable(lines.long(), requires_grad=False)
        target = Variable(target.long(), requires_grad=False)
        if use_cuda:
            lines, target = lines.cuda(), target.cuda()

        encoder_opt.zero_grad()  # clear gradients
        decoder_opt.zero_grad()
        encoder_hidden = repackage_hidden(encoder_hidden)
        decoder_hidden = repackage_hidden(decoder_hidden)

        all_hiddens, hidden = encoder(lines, encoder_hidden, xlens, dataset.max_len)

        # Grab final hidden state of encoder
        bi_hidden_to_uni(hidden)
        if encoder.rnn_type == 'LSTM':
            hidden = hidden[0]
        # Set to decoder's initial hidden state
        if decoder.rnn_type == 'GRU':
            decoder_hidden = hidden
        elif decoder.rnn_type == 'LSTM':
            decoder_hidden[0].data = hidden.data

        # first input to decoder is the eos token
        decoder_output = Variable(dataset.eos_tensor(batch_size, use_cuda), requires_grad=False)

        for i in range(sentence_len):  # process one word at a time per batch
            decoder_output, decoder_hidden, _ = decoder(decoder_output, decoder_hidden, all_hiddens)
            batch_loss += loss(decoder_output, target[:, i])
            _, idx = torch.max(decoder_output, 1)

            # batched per token teacher forcing
            if choices[i] == 1:
                decoder_output = target[:, i]
            else:
                decoder_output = idx  # set current prediction as decoder input

        batch_loss.backward()
        # clip gradients to 5 (hyper-parameter!) to reduce exploding gradients
        torch.nn.utils.clip_grad_norm(encoder.parameters(), 5)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), 5)
        encoder_opt.step()
        decoder_opt.step()
        total_loss += batch_loss

    return total_loss.data[0] / batch_size


def evaluate(encoder, decoder, batch_size, batches, dataset, loss):
    # turn off train mode to deactivate dropout between layers
    encoder.eval()
    decoder.eval()
    total_loss = 0
    use_cuda = encoder.is_cuda()
    encoder_hidden = encoder.init_hidden(batch_size)
    decoder_hidden = decoder.init_hidden(batch_size)

    for lines, target, xlens in batches:
        batch_loss = 0
        sentence_len = target.size()[1]
        lines = Variable(lines.long(), volatile=True)
        target = Variable(target.long(), volatile=True)
        if use_cuda:
            lines, target = lines.cuda(), target.cuda()

        # clear hidden state
        encoder_hidden = repackage_hidden(encoder_hidden)
        decoder_hidden = repackage_hidden(decoder_hidden)

        all_encoder_hiddens, hidden = encoder(lines, encoder_hidden,
                                              xlens, dataset.max_len)

        # Grab final hidden state of encoder
        bi_hidden_to_uni(hidden)
        if encoder.rnn_type == 'LSTM':
            hidden = hidden[0]
        # Set to decoder's initial hidden state
        if decoder.rnn_type == 'GRU':
            decoder_hidden = hidden
        elif decoder.rnn_type == 'LSTM':
            decoder_hidden[0].data = hidden.data

        decoder_output = Variable(dataset.eos_tensor(batch_size, use_cuda), volatile=True)

        for i in range(sentence_len):
            decoder_output, decoder_hidden, _ = decoder(decoder_output, decoder_hidden,
                                                        all_encoder_hiddens)
            batch_loss += loss(decoder_output, target[:, i])
            _, idx = torch.max(decoder_output, 1)
            decoder_output = idx  # feed current prediction as input to decoder
        # update total_loss
        total_loss += batch_loss

    return total_loss.data[0] / batch_size


def respond(encoder, decoder, input_line, dataset, input_len=None):
    """
    Generate chat response to user input, input_line.
    If input_len is given, skip sentence to embedding index conversion.
    """
    encoder.eval()
    decoder.eval()
    # turn off cuda
    encoder.cpu()
    decoder.cpu()
    # batch sizes are always 1
    encoder_hidden = encoder.init_hidden(1)
    decoder_hidden = decoder.init_hidden(1)

    if input_len is None:
        # translate input_line to indexes
        line = process_sentence(input_line, dataset.max_len)
        if line is None:  # input too long for model
            raise UserInputTooLongError
        else:
            input_len = [len(line)]
        input_indexes = torch.LongTensor(dataset.max_len).fill_(dataset.pad_idx)
        sentence_to_index(line, dataset, input_indexes)
    else:  # sentence is already in embedding index form
        input_indexes = input_line.long()
    input_line = Variable(input_indexes, volatile=True)

    all_encoder_hiddens, hidden = encoder(input_line.view(1, -1),
                                          encoder_hidden, input_len, dataset.max_len)

    # Grab final hidden state of encoder
    bi_hidden_to_uni(hidden)
    if encoder.rnn_type == 'LSTM':
        hidden = hidden[0]
    # Set to decoder's initial hidden state to final hidden of encoder
    if decoder.rnn_type == 'GRU':
        decoder_hidden = hidden
    elif decoder.rnn_type == 'LSTM':
        decoder_hidden[0].data = hidden.data

    decoder_output = Variable(dataset.eos_tensor(1, False), volatile=True)

    response = []
    for i in range(dataset.max_len):
        decoder_output, decoder_hidden, max_atten = decoder(decoder_output, decoder_hidden,
                                                            all_encoder_hiddens)
        _, idx = torch.max(decoder_output, 1)
        # Unknown word replacement:
        # replace unknown words with input word with highest attention
        if idx.data[0] == dataset.unk_idx and max_atten.data[0] <= input_len[0]:
            idx.data[0] = input_indexes[max_atten.data[0]]
        decoder_output = idx  # feed current prediction as input to decoder
        response.append(idx.data[0])

    # decode indexes to words
    response_sentence = []
    for i in range(len(response)):
        response_word = dataset.vocab.itos[response[i]]
        if response_word == dataset.eos_token:  # model output eos
            break
        else:
            response_sentence.append(response_word)

    return ' '.join(response_sentence)


def bi_hidden_to_uni(hidden):
    """
    Concatenate hidden states of bidirectional RNN
    to use as hidden state for unidirectional RNN.
    """
    hidden_list = []
    if isinstance(hidden, tuple):
        n_iter = hidden[0].size()[0] // 2
        for i in range(n_iter):
            hidden_list.append(torch.cat((hidden[0][i * 2], hidden[0][i * 2 + 1]), 1))
        hidden[0].data = torch.stack(hidden_list, 0).data
    else:
        n_iter = hidden.size()[0] // 2
        for i in range(n_iter):
            hidden_list.append(torch.cat((hidden[i * 2], hidden[i * 2 + 1]), 1))
        hidden.data = torch.stack(hidden_list, 0).data


def get_input(input_line, dataset):
    """
    Turn line of embedding indexes to human readable words.
    """
    sentence = []
    for i in range(input_line.size()[0]):
        idx = input_line[i]
        if idx == dataset.pad_idx:
            break
        word = dataset.vocab.itos[idx]
        sentence.append(word)
    return ' '.join(sentence)


def repackage_hidden(h):
    """
    Clear history of hidden state of rnn.
    """
    if type(h) == Variable:
        return Variable(h.data, requires_grad=False)
    else:
        return tuple(repackage_hidden(v) for v in h)


def plot(xvals, yvals, xlabel, ylabel):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xvals, yvals, 'r--')


class UserInputTooLongError(Exception):
    """
    User input is longer than the maximum length
    encoder and decoder are trained to handle.
    """
    pass
