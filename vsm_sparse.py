#!/usr/bin/python3
from scipy import sparse
import pickle
import numpy as np
import os
import math
from collections import defaultdict
from query_sparse import Query
total_doc = 46972
total_grams = 1193467
def save_obj(obj, name):
    with open(name, 'wb') as fp:
        pickle.dump(obj, fp, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open(name, 'rb') as fp:
        return pickle.load(fp)
class VSM:
    def __init__(self, invert_file_path, file_list):
        self.invert_file_path = invert_file_path
        self.file_list = file_list

        self.data = []
        self.tf_idf = None

        self.length = np.zeros(total_doc)
        self.idf = np.zeros(total_grams)
        self.index = dict()
        self.queries = None
        self.file_index = None
        with open(self.file_list) as fp:
            self.file_index = [line.strip().split('/')[-1].lower() for line in fp]
    def weight(self, tf_func=lambda tf, mtf: 0.5 * (float(tf)/float(mtf))):
        out_data = '/tmp3/ralph831005/IR/coo_weight.npy'
        out_index = '/tmp3/ralph831005/IR/index.binary'
        out_length = '/tmp3/ralph831005/IR/length.npy'
        if os.path.exists(out_data):
            self.data = np.load(out_data)
            self.index = load_obj(out_index)
            self.length = np.load(out_length)
            self.tf_idf = sparse.csr_matrix((self.data[2], (self.data[0], self.data[1])), shape=(total_doc, total_grams))
            return 

        with open(self.invert_file_path, 'r') as inverted_file:
            counter = 0
            while True:
                line = inverted_file.readline()
                if len(line) == 0:
                    break
                bigram = line.strip().split()
                self.index[int(bigram[0])*100000 + int(bigram[1])] = counter
                self.idf[counter] = math.log(total_doc/float(bigram[2]))
                lines = [[int(x) for x in inverted_file.readline().strip().split()] for i in range(int(bigram[2]))]
                max_tf = max(list(zip(*lines))[1])
                for doc in lines:
                    #self.tf_idf[doc[0]][counter] = tf_func(doc[1], max_tf) * self.idf[counter]
                    self.data.append((doc[0], counter, tf_func(doc[1], max_tf) * self.idf[counter]))
                    self.length[doc[0]] += (tf_func(doc[1], max_tf) * self.idf[counter])**2
                counter += 1
        for i in range(len(self.length)):
            self.length[i] = (self.length[i]**(0.5))
        self.data = np.transpose(self.data)
        np.save(out_data, self.data)
        np.save(out_length, self.length)
        save_obj(self.index, out_index)
        self.tf_idf = sparse.csr_matrix((self.data[2], (self.data[0], self.data[1])), shape=(total_doc, total_grams))
    def cosine_similarity(self, query):
        similarity = np.transpose(self.tf_idf.dot(query.sparse).toarray())[0]
        for i in range(total_doc):
            if self.length[i] == 0:
                similarity[i] = 0
            else:
                similarity[i] = similarity[i]/(self.length[i])/(query.length)
        return sorted(list(zip(range(total_doc), similarity)), key=lambda x: x[1], reverse=True)
    def rocchio_feedback(self, query, alpha, beta, pseudo_threshold = 10):
        adjust = sparse.dok_matrix((1, total_grams))
        for result in self.cosine_similarity(query)[:pseudo_threshold]:
            adjust = adjust+self.tf_idf[result[0]]
        query.sparse = ((alpha * query.sparse.transpose()) + ((beta / pseudo_threshold) * adjust)).transpose()
    def rank(self, output):
        with open(output, 'w') as fp:
            for query in self.queries:
                self.rocchio_feedback(query, 0.9, 0.2)
                for result in self.cosine_similarity(query)[:100]:
                    fp.write(query.number+' '+self.file_index[result[0]]+'\n')

    def parse(self, query_path, vocab_index):
        self.queries = Query.parse(query_path, vocab_index, self.index)
