#!/bin/python3
import numpy as np
import os
import math
invert_file_path = '/tmp2/Ralph/IR/model/inverted-file'
total_doc = 46972

def weight(file_path=invert_file_path, tf_func=lambda tf, mtf: 0.5 + 0.5 * (float(tf)/float(mtf))):
    tf_idf = [[] for i in range(total_doc)]
    index = []
    with open(file_path, 'r') as inverted_file:
        while True:
            line = inverted_file.readline()
            if len(line) == 0:
                break
            bigram = line.strip().split()
            index.append(bigram[0:2])
            idf = math.log(total_doc/float(bigram[2]))
            lines = dict([tuple([int(x) for x in inverted_file.readline().strip().split()]) for i in range(int(bigram[2]))])
            max_tf = max(lines.values())
            for i in range(total_doc):
                if i in lines:
                    tf_idf[i].append(tf_func(lines[i], max_tf) * idf)
                else:
                    tf_idf[i].append(tf_func(0, max_tf) * idf)
    return tf_idf, index

if __name__ == '__main__':
    print(weight('./test.in')[0][33689])

