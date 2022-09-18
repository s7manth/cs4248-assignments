# python3.8 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import re
import numpy as np
from collections import OrderedDict

START_WORD = '<~S>'
ALPHA = 1

def read_sentences(filename) : 
    with open(filename, 'r') as f : 
        sentences = f.read().strip().split("\n")
    
    return sentences

def train_model(train_file, model_file) : 
    sentences = read_sentences(train_file)
    data = list()

    words_as_set = set()
    tags_as_set = set()

    word_to_index = OrderedDict()
    tag_to_index = OrderedDict()

    tag_to_index[START_WORD] = 0

    word_to_index_length = 0
    tag_to_index_length = 1

    transition_matrix = np.zeros([46, 45], dtype=np.float32)

    for s in sentences : 
        ps = s.split(' ')
        ts = [x.rsplit('/', 1)[1] for x in ps]

        for p in ps : 
            p = p.rsplit('/', 1)
            for w in p[0].split('/') : 
                data.append((w, p[1]))

                if not w in word_to_index : 
                    word_to_index[w] = word_to_index_length
                    word_to_index_length += 1
                
                if not p[1] in tag_to_index : 
                    tag_to_index[p[1]] = tag_to_index_length
                    tag_to_index_length += 1
        
        ts.insert(0, START_WORD)
        for j in range(len(ts) - 1) : 
            transition_matrix[tag_to_index[ts[j]], tag_to_index[ts[j + 1]] - 1] += 1

    emission_matrix = np.zeros([word_to_index_length, 45], dtype=np.float32)
    tag_freq = np.zeros([1, 45], dtype=np.float32)

    for x in data : 
        emission_matrix[word_to_index[x[0]], tag_to_index[x[1]] - 1] += 1
        tag_freq[0, tag_to_index[x[1]] - 1] += 1

    transition_matrix += 1 * ALPHA

    transition_matrix /= np.matrix(transition_matrix).sum(axis=1)
    emission_matrix /= tag_freq

    with open(model_file, 'a') as f :
        np.savetxt(f, transition_matrix, header='transition matrix', newline='\n', footer='end transition matrix')
        np.savetxt(f, emission_matrix, header='emission matrix', newline='\n', footer='end emission matrix')
        np.savetxt(f, np.array([list(tag_to_index.keys())]), header='tags', fmt='%s', newline='\n', footer='end tags')
        np.savetxt(f, np.array([list(word_to_index.keys())]), header='words', fmt='%s', newline='\n', footer='end words')
        np.savetxt(f, tag_freq, header='tag frequency', newline='\n', footer='tag frequency')
    
    print('Finished...')

if __name__ == '__main__':
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
