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


def process_sentence(sentence, counter=None):
    """
    Lowercase, trim, and remove non-letter characters,
    and catalogue word counts in counter, if provided.
    """
    new_sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
    words = new_sentence.lower().strip().split(' ')
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
        input_line = process_sentence(input_line, word_count)
        target_line = sentences[i * 2 + 1][:-1]
        target_line = process_sentence(target_line, word_count)

        input_len = len(input_line) + 1  # + 1 for eos token
        target_len = len(target_line) + 1
        if input_len <= max_len and target_len <= max_len:
            input_data.append(input_line)
            target_data.append(target_line)
            input_lens.append(input_len)

    assert len(input_data) == len(target_data)
    return input_data, target_data, word_count, input_lens


def convert_to_index(data, vocab, unk_token, eos_idx, result):
    """
    Convert tokens in data to indexes, with indexing given by vocab.
    """
    n_lines = len(data)
    with ProcessPoolExecutor() as executer:
        for i in tnrange(n_lines, desc='Converting', unit=' lines'):
            sentence_to_index(data[i], vocab, unk_token, eos_idx, result[i])


def sentence_to_index(sentence, vocab, unk_token, eos_idx, output):
    """
    Convert list of words to tensor of indexes.
    Precondition: len(sentence) < output.shape[0]
    """
    assert len(sentence) < output.shape[0]
    for i in range(len(sentence)):
        word = sentence[i]
        idx = vocab.stoi[word]
        if idx == 0 and word != vocab.itos[0]:
            idx = vocab.stoi[unk_token]  # unknown word encountered
        output[i] = idx
    output[len(sentence)] = eos_idx
