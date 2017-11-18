import torch
import torch.nn as nn
import numpy as np
from random import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_utils import process_sentence, sentence_to_index


def train(encoder, decoder, batch_size, batch, encoder_opt, decoder_opt,
          params, dataset, choices, loss, drop_t, drop_w):
    # turn on train mode to activate dropout between layers
    encoder.train()
    decoder.train()
    use_cuda = encoder.is_cuda()
    hidden = encoder.init_hidden(batch_size)
    decoder_hidden = decoder.init_hidden(batch_size)

    # Create dropout masks
    word_drop_probs = torch.zeros(dataset.nwords).fill_(1 - drop_w)
    if encoder.rnn_type == 'GRU':
        dropout_probs = torch.zeros(hidden.size()).fill_(1 - drop_t)
    else:
        dropout_probs = torch.zeros(hidden[0].size()).fill_(1 - drop_t)

    word_mask = torch.bernoulli(word_drop_probs).long()
    masked_indexes = torch.zeros(batch_size).long()
    drop_mask = torch.bernoulli(dropout_probs)
    drop_scale = 1 / (1 - drop_t)

    # Container for encoder inputs and all top encoder hidden states
    input = Variable(torch.zeros(dataset.nwords), requires_grad=False)
    all_hiddens = Variable(torch.zeros(batch_size, dataset.max_len,
                                       encoder.hidden_size), requires_grad=False)

    lines, target, _ = batch
    sentence_len = target.size()[1]
    batch_loss = 0

    lines = Variable(lines.long(), requires_grad=False)
    target = Variable(target.long(), requires_grad=False)
    if use_cuda:
        lines, target = lines.cuda(), target.cuda()
        word_mask, drop_mask = word_mask.cuda(), drop_mask.cuda()
        input, all_hiddens = input.cuda(), all_hiddens.cuda()
        masked_indexes = masked_indexes.cuda()

    encoder_opt.zero_grad()  # clear gradients
    decoder_opt.zero_grad()

    for i in range(sentence_len):
        # word dropout: drop the same word randomly
        masked_indexes = word_mask.index(lines[:, i].data)
        input.data = torch.mul(masked_indexes, lines[:, i].data)

        all_hiddens[:, i], hidden = encoder(input, hidden)
        # Varational Dropout (dropout with same mask at each time step)
        if encoder.rnn_type == 'GRU':
            hidden.data = torch.mul(drop_mask, hidden.data) * drop_scale
        elif encoder.rnn_type == 'LSTM':
            hidden[0].data = torch.mul(drop_mask, hidden[0].data) * drop_scale

    # reshape drop_mask for use in decoder
    drop_mask = torch.cat(drop_mask, 1).view(encoder.nlayers, batch_size,
                                             encoder.hidden_size)
    # Grab final hidden state of encoder
    if encoder.rnn_type == 'GRU':
        hidden.data = torch.cat(hidden, 1).view(encoder.nlayers, batch_size,
                                                encoder.hidden_size).data
    elif encoder.rnn_type == 'LSTM':
        hidden[0].data = torch.cat(hidden[0], 1).view(encoder.nlayers, batch_size,
                                                      encoder.hidden_size).data
        hidden = hidden[0]

    # Set to decoder's initial hidden state
    if decoder.rnn_type == 'GRU':
        decoder_hidden = hidden
    elif decoder.rnn_type == 'LSTM':
        decoder_hidden[0].data = hidden.data

    # start-of-string
    decoder_output = Variable(dataset.sos_tensor(batch_size, use_cuda), requires_grad=False)

    for i in range(sentence_len):  # process one word at a time per batch

        decoder_output, decoder_hidden, _ = decoder(decoder_output, decoder_hidden, all_hiddens)
        batch_loss += loss(decoder_output, target[:, i])
        _, idx = torch.max(decoder_output, 1)

        # batched per token teacher forcing
        if choices[i] == 1:
            decoder_output = target[:, i]
        else:
            decoder_output = idx  # set current prediction as decoder input

        # Varational Dropout (dropout with same mask at each time step)
        if decoder.rnn_type == 'GRU':
            decoder_hidden.data = torch.mul(drop_mask, decoder_hidden.data) * drop_scale
        elif decoder.rnn_type == 'LSTM':
            decoder_hidden[0].data = torch.mul(drop_mask, decoder_hidden[0].data) * drop_scale

    batch_loss.backward()
    # clip gradients to 5 (hyper-parameter!) to reduce exploding gradients
    torch.nn.utils.clip_grad_norm(encoder.parameters(), 5)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), 5)
    encoder_opt.step()
    decoder_opt.step()

    return batch_loss.data[0] / batch_size


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
        if encoder.rnn_type == 'GRU':
            hidden.data = torch.cat(hidden, 1).view(encoder.nlayers, batch_size,
                                                    encoder.hidden_size).data
        elif encoder.rnn_type == 'LSTM':
            hidden[0].data = torch.cat(hidden[0], 1).view(encoder.nlayers, batch_size,
                                                          encoder.hidden_size).data
            hidden = hidden[0]
        # Set to decoder's initial hidden state
        if decoder.rnn_type == 'GRU':
            decoder_hidden = hidden
        elif decoder.rnn_type == 'LSTM':
            decoder_hidden[0].data = hidden.data

        # start-of-string
        decoder_output = Variable(dataset.sos_tensor(batch_size, use_cuda), volatile=True)

        for i in range(sentence_len):
            decoder_output, decoder_hidden, _ = decoder(decoder_output, decoder_hidden,
                                                        all_encoder_hiddens)
            batch_loss += loss(decoder_output, target[:, i])
            _, idx = torch.max(decoder_output, 1)
            decoder_output = idx  # feed current prediction as input to decoder
        # update total_loss
        total_loss += (batch_loss / batch_size)

    return total_loss.data[0]


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
    if encoder.rnn_type == 'GRU':
        hidden.data = torch.cat(hidden, 1).view(encoder.nlayers, 1,
                                                encoder.hidden_size).data
    elif encoder.rnn_type == 'LSTM':
        hidden[0].data = torch.cat(hidden[0], 1).view(encoder.nlayers, 1,
                                                      encoder.hidden_size).data
        hidden = hidden[0]
    # Set to decoder's initial hidden state
    if decoder.rnn_type == 'GRU':
        decoder_hidden = hidden
    elif decoder.rnn_type == 'LSTM':
        decoder_hidden[0].data = hidden.data

    # start-of-string
    decoder_output = Variable(dataset.sos_tensor(1, False), volatile=True)

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
