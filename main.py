#!/usr/bin/python3
import numpy as np
import os
import math
import sys
from collections import defaultdict
from vsm_sparse import VSM

def parse_command():
    command = dict()
    if '-r' in sys.argv:
        command['-r'] = True
    else:
        command['-r'] = False
    outfile = sys.argv.index('-o')
    command['-o'] = sys.argv[outfile+1]
    outfile = sys.argv.index('-i')
    command['-i'] = sys.argv[outfile+1]
    outfile = sys.argv.index('-m')
    command['-m'] = sys.argv[outfile+1]
    outfile = sys.argv.index('-d')
    command['-d'] = sys.argv[outfile+1]
    return command

def main():
    vocab_index = dict()
    command = parse_command()
    train_path = '/tmp3/ralph831005/IR/query/query-train.xml'
    ans = '/tmp3/ralph831005/IR/query/ans-train'
    output_train = 'simple_train_vsm.txt'
    invert_file_path = command['-d'] + 'inverted-file'
    file_list = command['-d'] + 'file-list'
    vocal_all = command['-d'] + 'vocab.all'
    output_test = command['-o']
    test_path = command['-i']
    with open(vocal_all) as fp:
        for i, vocab in enumerate(fp):
            vocab_index[vocab.strip()] = i

    vsm = VSM(invert_file_path, file_list, command['-m'])
    #vsm = VSM('test.in', file_list)
    vsm.weight()
    vsm.parse(train_path, vocab_index)
    vsm.rank(output_train, command['-r'], lsi = False, alpha=0.9, beta=0.1, pseudo_threshold=3)
    vsm.parse(test_path, vocab_index)
    vsm.rank(output_test, command['-r'], lsi = False, alpha=0.9, beta=0.1, pseudo_threshold=3)

    '''
    for i in range(1, 10):
        vsm.parse(train_path, vocab_index)
        vsm.rank('train_rocchio_a_0.9_b_0.1_k_'+str(i)+'.txt', command['-r'], lsi=False, alpha=0.9, beta=0.1, pseudo_threshold=i)
        print(eval(ans, 'train_rocchio_a_0.9_b_0.1_k_'+str(i)+'.txt'))
        vsm.parse(test_path, vocab_index)
        vsm.rank('test_rocchio_a_0.9_b_0.1_k_'+str(i)+'.txt', command['-r'], lsi=False, alpha=0.9, beta=0.1, pseudo_threshold=i)
        '''
def apk(actual, predicted):
    k = min(100, len(predicted))
    score, count = 0, 0
    for i, pred in enumerate(predicted):
        if pred in actual:
            count += 1
            score += count/(i+1.0)
    return score/min(len(actual), k)
def eval(ans_path, pred_path):
    ans_set = defaultdict(set)
    pred_set = defaultdict(list)
    with open(ans_path) as fp:
        for tmp in fp:
            line = tmp.strip().split()
            ans_set[line[0]].add(line[1])
    with open(pred_path) as fp:
        for tmp in fp:
            line = tmp.strip().split()
            pred_set[line[0]].append(line[1].lower().split('/')[-1])
    return np.mean([apk(ans_set[k], pred_set[k]) for k in ans_set.keys()])

if __name__ == '__main__':
    main()
