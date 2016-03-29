#!/bin/python3
import numpy as np
import os
import math
invert_file_path = '/tmp2/Ralph/IR/model/inverted-file'
total_doc = 46972

class tf_idf:
    def __init__(self, file_path=invert_file_path, tf_func=lambda tf, mtf: 0.5 + 0.5(float(tf)/float(mtf))):
        self.tf_idf = [0 for i in range(total_doc)]
        self.index = []
        with open(file_path, 'r') as inverted_file:
            while True:
                line = inverted_file.readline()
                if len(line) == 0:
                    break
                bigram = line.strip().split()
                self.index.append(bigram[0:2])
                idf = math.log(float(bigram[2])/total_doc)
                lines = [[int(x) for x in inverted_file.readline().strip().split()] for i in range(int(bigram[2]))]
                max_tf = max(list(zip(*lines))[1])


                
                    

    @staticmethod
    def load(file_path):


