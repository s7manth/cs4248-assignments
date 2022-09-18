# python3.8 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
from itertools import islice
from collections import OrderedDict
import numpy as np
import cProfile

START_WORD = '<~S>'

def read_model(model_file) : 
    line_ranges = list()

    with open(model_file, 'r') as f : 
        for l_num, line in enumerate(f) : 
            if line.strip() == '# transition matrix' : 
                line_ranges.append(l_num + 1)
            
            if line.strip() == '# end transition matrix' : 
                line_ranges.append(l_num + 1)
            
            if line.strip() == '# emission matrix' : 
                line_ranges.append(l_num + 1)

            if line.strip() == '# end emission matrix' : 
                line_ranges.append(l_num + 1)

            if line.strip() == '# tags' : 
                line_ranges.append(l_num + 1)
            
            if line.strip() == '# words' : 
                line_ranges.append(l_num + 1)

            if line.strip() == '# tag frequency' : 
                line_ranges.append(l_num + 1)

    arrs = list()

    for i in range(0, 4, 2) : 
        with open(model_file) as f : 
            arrs.append(np.loadtxt(islice(f, line_ranges[i], line_ranges[i + 1])))

    for i in range(4, 7) : 
        with open(model_file) as f : 
            arrs.append(np.array((f.readlines()[line_ranges[i]]).strip().split(' ')))

    return arrs


def pos_tagged(sentence, transition_matrix, emission_matrix, tag_to_index, word_to_index, tags, out_file) : 
    words_in_a_sentence = sentence.strip().split(' ')
    len_sentence = len(words_in_a_sentence)
    num_tags = len(tags)

    prob_dict = np.zeros([len_sentence, num_tags])
    tag_dict = np.zeros([len_sentence, num_tags])

    for i in range(num_tags) :
        prob_dict[0, i] = transition_matrix[tag_to_index[START_WORD], i - 1]

        if words_in_a_sentence[0] in word_to_index : 
            prob_dict[0, i] *= emission_matrix[word_to_index[words_in_a_sentence[0]], i - 1]
        else : 
            prob_dict[0, i] *= 0.01

        tag_dict[0, i] = -1
    
    for i in range(1, len_sentence) : 
        for j in range(num_tags) : 
            if words_in_a_sentence[i] in word_to_index : 
                em_p = emission_matrix[word_to_index[words_in_a_sentence[i]], j - 1] 
            else : 
                em_p = 0.01
            
            xs = [prob_dict[i - 1, k] * transition_matrix[k, j - 1] * em_p for k in range(num_tags)]
            prob_dict[i, j] = max(xs)
            tag_dict[i, j] = np.argmax(np.array(xs))

    t_max = int(np.argmax(prob_dict[-1, :]))
    vit_max = prob_dict[-1, t_max]

    sent_out = list()

    i = len_sentence - 1

    while i >= 0 : 
        sent_out.append(words_in_a_sentence[i] + '/' + tags[t_max])
        t_max = int(tag_dict[i, t_max])
        i -= 1

    sent_out.reverse()

    return ' '.join(sent_out) + '\n'

def tag_sentence(test_file, model_file, out_file) :
    transition_matrix, emission_matrix, tags, words, tag_freq = read_model(model_file)

    tag_to_index = OrderedDict()
    for i, t in enumerate(tags) : 
        tag_to_index[t] = i

    word_to_index = OrderedDict()
    for i, w in enumerate(words) : 
        word_to_index[w] = i

    result = list()

    with open(test_file, 'r') as f : 
        sentences = f.read().strip().split("\n")

    rs = [pos_tagged(s, transition_matrix, emission_matrix, tag_to_index, word_to_index, tags, out_file) for s in sentences]

    with open(out_file, 'w') as f : 
        f.writelines(rs)

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    cProfile.run("tag_sentence(test_file, model_file, out_file)")
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
