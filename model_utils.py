import torch
import torch.nn as nn
import numpy as np
from random import random
from sympy import exp, Symbol
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import collate_fn
from data_utils import process_sentence, sentence_to_index


def train(encoder, decoder, batch_size, sampler, optimizer, params, dataset,
          choices, loss, e_dropout, d_dropout):
    batch_loss = 0
    # Do not calculate loss when target is padding
    # Otherwise, we discourage model from saying more than target length
    use_cuda = encoder.is_cuda()
    encoder_hidden = encoder.init_hidden(batch_size)
    decoder_hidden = decoder.init_hidden(batch_size)
    # Create dropout masks
    if encoder.rnn_type == 'GRU':
        e_dropout_probs = torch.zeros(encoder_hidden.size()).fill_(e_dropout)
        e_zero_tensor = torch.zeros(encoder_hidden.size())
    else:
        e_dropout_probs = torch.zeros(encoder_hidden[0].size()).fill_(e_dropout)
        e_zero_tensor = torch.zeros(encoder_hidden[0].size())
    if decoder.rnn_type == 'GRU':
        d_dropout_probs = torch.zeros(decoder_hidden.size()).fill_(d_dropout)
        d_zero_tensor = torch.zeros(decoder_hidden.size())
    else:
        d_dropout_probs = torch.zeros(decoder_hidden[0].size()).fill_(d_dropout)
        d_zero_tensor = torch.zeros(decoder_hidden[0].size())

    e_mask = torch.bernoulli(e_dropout_probs)
    d_mask = torch.bernoulli(d_dropout_probs)
    e_scale = 1 / (1 - e_dropout)
    d_scale = 1 / (1 - d_dropout)

    # Container for encoder outputs
    encoder_output = Variable(torch.zeros(batch_size, dataset.max_len,
                                          encoder.hidden_size), requires_grad=False)

    # Create new instance of DataLoader to simulate sampling with
    # replacement (PyTorch SubsetRandomSampler does not).
    # We are also willing to waste some memory by turning on pin_memory
    batches = DataLoader(dataset, batch_size=batch_size,
                         sampler=sampler, num_workers=0,
                         collate_fn=collate_fn, pin_memory=True,
                         drop_last=True)
    batch = next(iter(batches))  # get one batch from batches
    lines, target, xlens = batch
    sentence_len = target.size()[1]

    lines = Variable(lines.long(), requires_grad=False)
    target = Variable(target.long(), requires_grad=False)
    if use_cuda:
        lines, target = lines.cuda(), target.cuda()
        e_mask, d_mask = e_mask.cuda(), d_mask.cuda()
        e_zero_tensor, d_zero_tensor = e_zero_tensor.cuda(), d_zero_tensor.cuda()
        encoder_output = encoder_output.cuda()

    # clear hidden state
    encoder_hidden = repackage_hidden(encoder_hidden)
    decoder_hidden = repackage_hidden(decoder_hidden)

    optimizer.zero_grad()  # clear gradients

    for i in range(xlens[0]):
        encoder_output[:, i], encoder_hidden = encoder(lines[:, i], hidden)
        # Varational Dropout (dropout with same mask at each time step)
        if encoder.rnn_type == 'GRU':
            encoder_hidden.data = torch.addcmul(e_zero_tensor, e_scale,
                                                e_mask, encoder_hidden.data)
        elif encoder.rnn_type == 'LSTM':
            encoder_hidden[0].data = torch.addcmul(e_zero_tensor, e_scale,
                                                   e_mask, encoder_hidden[0].data)

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

    decoder_output = dataset.sos_tensor(batch_size, use_cuda)  # start-of-string

    for i in range(sentence_len):  # process one word at a time per batch
        decoder_output, decoder_hidden = decoder(decoder_output, decoder_hidden, output)
        batch_loss += loss(decoder_output, target[:, i])
        _, idx = torch.max(decoder_output, 1)

        # batched per token teacher forcing
        if choices[i] == 1:
            decoder_output = target[:, i]
        else:
            decoder_output = idx  # set current prediction as decoder input

        # Varational Dropout (dropout with same mask at each time step)
        if decoder.rnn_type == 'GRU':
            decoder_hidden.data = torch.addcmul(d_zero_tensor, d_scale,
                                                d_mask, decoder_hidden.data)
        elif decoder.rnn_type == 'LSTM':
            decoder_hidden[0].data = torch.addcmul(d_zero_tensor, d_scale,
                                                   d_mask, decoder_hidden[0].data)

    batch_loss.backward()
    # clip gradients to 0.5 (hyper-parameter!) to reduce exploding gradients
    torch.nn.utils.clip_grad_norm(params, 0.5)
    optimizer.step()

    del batches
    return batch_loss.data[0]


def evaluate(encoder, decoder, batch_size, batches, dataset, loss):
    total_loss = 0
    use_cuda = encoder.is_cuda()
    encoder_hidden = encoder.init_hidden(batch_size)
    decoder_hidden = decoder.init_hidden(batch_size)

    # Calculate validation loss on entire validation set
    for lines, target, xlens in batches:
        sentence_len = target.size()[1]
        lines = Variable(lines.long(), volatile=True)
        target = Variable(target.long(), volatile=True)
        if use_cuda:
            lines, target = lines.cuda(), target.cuda()

        # clear hidden state
        encoder_hidden = repackage_hidden(encoder_hidden)
        decoder_hidden = repackage_hidden(decoder_hidden)

        output, hidden = encoder(lines, encoder_hidden, xlens, dataset.max_len)

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

        decoder_output = dataset.sos_tensor(batch_size, use_cuda)  # start-of-string

        for i in range(sentence_len):
            decoder_output, decoder_hidden = decoder(decoder_output, decoder_hidden, output)
            total_loss += loss(decoder_output, target[:, i])
            _, idx = torch.max(decoder_output, 1)
            decoder_output = idx  # feed current prediction as input to decoder

    return total_loss.data[0] / len(batches)


def respond(encoder, decoder, input_line, dataset):
    """
    Generate chat response to user input, input_line.
    """
    # turn off cuda
    encoder.cpu()
    decoder.cpu()

    encoder_hidden = encoder.init_hidden(1)
    decoder_hidden = decoder.init_hidden(1)

    # translate input_line to indexes
    line = process_sentence(input_line)
    line_len = len(line) + 1  # + 1 for eos token
    if line_len > dataset.max_len:  # input too long for model
        raise UserInputTooLongError
    input_indexes = torch.LongTensor(dataset.max_len).fill_(dataset.pad_idx)
    sentence_to_index(line, dataset.vocab, dataset.unk_token, dataset.eos_idx, input_indexes)
    input_line = Variable(input_indexes, volatile=True)

    output, hidden = encoder(input_line.view(1, -1), encoder_hidden, [line_len], dataset.max_len)

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

    decoder_output = dataset.sos_tensor(1, False)  # start-of-string

    response = []
    for i in range(dataset.max_len):
        decoder_output, decoder_hidden = decoder(decoder_output, decoder_hidden, output)
        _, idx = torch.max(decoder_output, 1)
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


def repackage_hidden(h):
    """
    Clear history of hidden state of rnn.
    """
    if type(h) == Variable:
        return Variable(h.data, requires_grad=False)
    else:
        return tuple(repackage_hidden(v) for v in h)


def inverse_sigmoid_decay(x, k):
    """
    Symbolic calculation of inverse_sigmoid.
    The speed of convergence to 0 is given by k.
    Lower values of k converge faster.
    Precondition: k >= 1
    """
    return k / (k + exp(x / k))


def inv_sigm_eval(x_val, k_val):
    """
    Evaluation of inverse sigmoid decay.

    Args:
        x_val (int): value to substitute for Symbol x
        k_val (int): value to substitute for Symbol k

    """
    x = Symbol('x')
    k = Symbol('k')
    k_val = 1 if k_val < 1 else k_val
    expr = inverse_sigmoid_decay(x, k)
    res = expr.subs([(x, x_val), (k, k_val)])
    return res.evalf()


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
