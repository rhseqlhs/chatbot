import os
from os import path
from urllib.request import urlopen
import shutil
import gzip
import re
import numpy as np
import torch
import math
import time
from tqdm import tnrange
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import random

# Helper functions for initializing data


def download(url, filename_list, data_dir):
    """
    Download data from url to data_dir, unless a file in filename_list exist.
    Precondition: filename_list must contain desired file as its first element.
    """
    if len(filename_list) == 0:
        print("Please provide name of desired file as second argument.")
        print("e.g. download('https://...f_name', [f_name], DATA_DIR")

    for filename in filename_list:
        filename = path.join(data_dir, filename)
        if path.exists(filename):
            print("No need to download %s" % filename_list[0])
            return

    download_dest = path.join(data_dir, filename_list[0])
    print("Downloading from %s" % url)
    with urlopen(url) as response, open(download_dest, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print("%s downloaded" % filename_list[0])


def concatenate_two_gz(file_path, suffix1, suffix2):
    """
    Given two .gz files whose paths are:
    file_path + suffix1, file_path + suffix2
    concatenate the two .gz files by appending file ending in suffix2
    to file ending in suffix1, only if concatenated file does not exist
    and unzipped file does not exist.
    """
    file_1 = file_path + suffix1
    file_2 = file_path + suffix2
    # check if concatenation is necessary
    if not path.exists(file_path[:-3]) and not path.exists(file_path):
        with open(file_1, 'ab') as f1, open(file_2, 'rb') as f2:
            f1.write(f2.read())
            print("Concatenation done.")
        os.rename(file_path + suffix1, file_path)  # rename to remove suffix
    else:
        print("Concatenation not necessary.")


def unzip_gz(file_name, data_dir):
    """
    Unzip .gz file, file_name whose directory is given by data_dir.
    Assumes we have write permission on data_dir.
    """
    file_path = path.join(data_dir, file_name)  # location of file to unzip
    dest_file_path = file_path[:-3]  # remove .gz
    if not path.exists(dest_file_path):  # check if unzipped file exists
        with gzip.open(file_path, 'rb') as s, open(dest_file_path, 'wb') as d:
            shutil.copyfileobj(s, d)
            print("%s decompressed." % file_name)
    else:
        print("%s already exists, decompression unneeded." % file_name[:-3])


def create_sample(source, dest, data_dir, n_lines):
    """
    Create a text file of name dest from n_lines number of lines
    of text file of name source.
    """
    if n_lines % 2 == 1:
        n_lines -= 1
    source_path = path.join(data_dir, source)
    dest_path = path.join(data_dir, dest)
    with open(source_path, 'r') as s, open(dest_path, 'w') as d:
        for i in range(n_lines):
            sentence = s.readline()
            if sentence == '':  # end of source file reached
                break
            d.write(sentence)
    print("%s created." % dest)


def process_sentence(sentence, max_len, counter=None):
    """
    Lowercase, trim, and remove non-letter characters,
    and catalogue word counts in counter, if provided.
    Should the number of words exceed max_len, return None.
    """
    # reduce number of exclamations, etc
    sentence = re.sub(r"([.!?])", r" \1", sentence)
    # remove non-letter characters
    new_sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
    words = new_sentence.lower().strip().split(' ')
    if len(words) > max_len:
        return None
    else:
        for word in words:
            if counter is not None:
                counter[word] += 1
        return words


def process_data(data_path, max_len, eos, pad):
    """
    Return input, target split, and word count information from data.
    Precondition: file given by data_path has even number of lines.
    """
    input_data = []
    target_data = []
    input_lens = []
    word_count = Counter()

    with open(data_path, encoding='utf-8') as d_file:
        lines = d_file.read()

    sentences = lines.split('\n')
    n_lines = len(sentences) - 1  # sentences[-1] is ''
    n_iter = n_lines // 2  # process two lines at a time

    for i in tnrange(n_iter, desc='Processing', unit=' lines'):
        input_line = sentences[i * 2][:-1]  # remove '\n' character
        input_line = process_sentence(input_line, max_len, word_count)
        target_line = sentences[i * 2 + 1][:-1]
        target_line = process_sentence(target_line, max_len, word_count)

        if input_line is not None and target_line is not None:
            input_data.append(input_line)
            target_data.append(target_line)
            input_lens.append(len(input_line))

    assert len(input_data) == len(target_data)
    return input_data, target_data, word_count, input_lens


def convert_to_index(data, dataset, result):
    """
    Convert tokens in data to indexes.
    """
    n_lines = len(data)
    with ProcessPoolExecutor() as executer:
        for i in tnrange(n_lines, desc='Converting', unit=' lines'):
            sentence_to_index(data[i], dataset, result[i])


def sentence_to_index(sentence, dataset, output):
    """
    Convert list of words to tensor of indexes.
    """
    for i in range(len(sentence)):
        word = sentence[i]
        idx = dataset.vocab.stoi[word]
        if idx == 0 and word != dataset.vocab.itos[0]:
            # unknown word encountered
            idx = dataset.unk_idx
        output[i] = idx

def prune_data(data, target, data_lens, dataset, threshold):
    """
    Remove sentences from data and target if unknown words
    occur more than threshold.
    """
    n_iter = target.shape[0]
    prune_list = []
    for i in tnrange(n_iter, desc='Pruning', unit=' lines'):
        data_unk_count = len(np.where(data[i] == dataset.unk_idx)[0])
        target_unk_count = len(np.where(target[i] == dataset.unk_idx)[0])
        if data_unk_count >= threshold or target_unk_count >= threshold:
            prune_list.append(i)
        else:  # update word counts
            dataset.unk_count += (data_unk_count + target_unk_count)
            # 0 is assumed to be padding
            dataset.total_tokens += (np.count_nonzero(data[i]) + np.count_nonzero(target[i]))
    # delete sentences in prune list
    data = np.delete(data, prune_list, 0)
    target = np.delete(target, prune_list, 0)
    new_data_lens = []
    for i in range(len(data_lens)):
        if i not in prune_list:
            new_data_lens.append(data_lens[i])
    assert len(new_data_lens) == data.shape[0]
    return data, target, new_data_lens
