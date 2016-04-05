#!/usr/bin/python3
import numpy as np
import os
import math
from collections import defaultdict
from query import Query
total_doc = 46972
total_grams = 1193467
class VSM:
    def __init__(self, invert_file_path, file_list):
        self.invert_file_path = invert_file_path
        self.file_list = file_list
        self.tf_idf = [dict() for i in range(total_doc)]
        self.v_length = np.zeros(total_doc)
        self.idf = np.zeros(total_grams)
        self.index = defaultdict(dict)
        self.queries = None
        self.file_index = None
        with open(self.file_list) as fp:
            self.file_index = [line.strip().split('/')[-1].lower() for line in fp]
    def weight(self, tf_func=lambda tf, mtf: 0.5 * (float(tf)/float(mtf))):
        with open(self.invert_file_path, 'r') as inverted_file:
            counter = 0
            while True:
                line = inverted_file.readline()
                if len(line) == 0:
                    break
                bigram = line.strip().split()
                self.index[int(bigram[0])][int(bigram[1])] = counter
                self.idf[counter] = math.log(total_doc/float(bigram[2]))
                lines = [[int(x) for x in inverted_file.readline().strip().split()] for i in range(int(bigram[2]))]
                max_tf = max(list(zip(*lines))[1])
                for doc in lines:
                    self.tf_idf[doc[0]][counter] = tf_func(doc[1], max_tf) * self.idf[counter]
                    self.v_length[doc[0]] += self.tf_idf[doc[0]][counter] ** 2
                counter += 1
        for i in range(len(self.v_length)):
            self.v_length[i] = (self.v_length[i]**(0.5))
    def cosine_similarity(self, query):
        similarity = [0 for _ in range(total_doc)]
        for i, doc in enumerate(self.tf_idf):
            for k, v in query.sparse.items():
                if k in doc:
                    similarity[i] += doc[k]*v
            similarity[i] = similarity[i]/(self.v_length[i])/(query.length)
        return sorted(list(zip(range(total_doc), similarity)), key=lambda x: x[1], reverse=True)
    def rank(self, output):
        with open(output, 'w') as fp:
            for query in self.queries:
                for result in self.cosine_similarity(query)[:100]:
                    fp.write(query.number+' '+self.file_index[result[0]]+'\n')

    def parse(self, query_path, vocab_index):
        self.queries = Query.parse(query_path, vocab_index, self.index)
