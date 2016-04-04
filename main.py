#!/usr/bin/python3
import numpy as np
import os
import math
import sys
from collections import defaultdict
from vsm import VSM

invert_file_path = '/tmp3/ralph831005/IR/model/inverted-file'
file_list = '/tmp3/ralph831005/IR/model/file-list'
vocal_all = '/tmp3/ralph831005/IR/model/vocab.all'
train_path = '/tmp3/ralph831005/IR/query/query-train.xml'
test_path = '/tmp3/ralph831005/IR/query/query-test.xml'
ans = '/tmp3/ralph831005/IR/query/ans-train'
output_train = 'simple_train_vsm.txt'
output_test = 'simple_test_vsm.txt'
def main():
    vocab_index = dict()
    with open(vocal_all) as fp:
        for i, vocab in enumerate(fp):
            vocab_index[vocab.strip()] = i

    vsm = VSM(invert_file_path, file_list)
    vsm.weight(tf_func = lambda tf, mtf: float(tf))
    vsm.parse(train_path, vocab_index)
    vsm.rank(output_train)
    vsm.parse(test_path, vocab_index)
    vsm.rank(output_test)
    eval(ans, output_train)
def apk(actual, predicted):
    k = min(100, len(predicted))
    score, count = 0, 0
    for i, pred in enumerate(predicted):
        if pred in actual:
            count += 1
            score += count/(i+1.0)
    print(score/min(len(actual), k))
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
