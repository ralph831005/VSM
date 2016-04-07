#!/usr/bin/python3
import pickle
import numpy as np
import os
import math
from collections import defaultdict
from query_sparse import Query
from scipy import sparse
import scipy.sparse.linalg as la
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

        self.tf_idf = None
        self.reduce_mapping = None
        self.sigma = None
        self.length = np.zeros(total_doc)
        self.idf = np.zeros(total_grams)
        self.index = dict()
        self.queries = None
        self.file_index = None
        with open(self.file_list) as fp:
            self.file_index = [line.strip().split('/')[-1].lower() for line in fp]
    def weight(self, tf_func=lambda tf, mtf: 0.5 * (float(tf)/float(mtf))):
        out_data = '/tmp2/Ralph/IR/coo_weight.npy'
        out_index = '/tmp2/Ralph/IR/index.binary'
        out_length = '/tmp2/Ralph/IR/length.npy'
        print('weighting start')
        if os.path.exists(out_data):
            self.index = load_obj(out_index)
            self.length = np.load(out_length)
            self.tf_idf = load_obj(out_data)
            print('weight cache read')
            return 

        data = []
        with open(self.invert_file_path, 'r') as inverted_file:
            counter = 0
            while True:
                line = inverted_file.readline()
                if len(line) == 0:
                    break
                bigram = line.strip().split()
                tmp = counter
                if (int(bigram[0])*100000 + int(bigram[1])) in  self.index:
                    tmp = self.index[int(bigram[0])*100000 + int(bigram[1])]
                self.index[int(bigram[0])*100000 + int(bigram[1])] = tmp
                self.idf[tmp] = math.log(total_doc/float(bigram[2]))
                lines = [[int(x) for x in inverted_file.readline().strip().split()] for i in range(int(bigram[2]))]
                max_tf = max(list(zip(*lines))[1])
                for doc in lines:
                    #self.tf_idf[doc[0]][tmp] = tf_func(doc[1], max_tf) * self.idf[tmp]
                    data.append((doc[0], tmp, tf_func(doc[1], max_tf) * self.idf[tmp]))
                    self.length[doc[0]] += (tf_func(doc[1], max_tf) * self.idf[tmp])**2
                counter += 1
        for i in range(len(self.length)):
            self.length[i] = (self.length[i]**(0.5))
        data = np.transpose(data)
        np.save(out_length, self.length)
        save_obj(self.index, out_index)
        self.tf_idf = sparse.csr_matrix((data[2], (data[0], data[1])), shape=(total_doc, total_grams))
        save_obj(self.tf_idf, out_data)
        print('weighting done')
    def lsi(self, n_sigular_value=200):
        print('lsi start')
        out_u = '/tmp2/Ralph/IR/u.npy'
        out_inv_sigma = '/tmp2/Ralph/IR/sig.npy'
        out_v = '/tmp2/Ralph/IR/v.npy'
        if os.path.exists(out_v) and os.path.exists(out_inv_sigma) and os.path.exists(out_u):
            self.tf_idf = np.load(out_u)
            self.reduce_mapping = np.load(out_v)
            self.sigma = np.load(out_inv_sigma)
            print('lsi cache read')
            return
        self.tf_idf, sigma, v = la.svds(self.tf_idf, k = n_sigular_value)
        self.reduce_mapping = v.transpose()
        self.sigma = la.inv(sparse.csr_matrix((sigma, (range(n_sigular_value), range(n_sigular_value))), shape=(n_sigular_value, n_sigular_value))).toarray()
        np.save(out_u, self.tf_idf)
        np.save(out_inv_sigma, self.sigma)
        np.save(out_v, self.reduce_mapping)
        print('lsi done')
    def cosine_similarity(self, query):
        similarity = np.transpose(sparse.csr_matrix(self.tf_idf.dot(query.sparse)).toarray())[0]
        '''
        try:
            similarity = np.transpose(similarity.toarray())[0]
        except:
            print('in')
            similarity = np.transpose(similarity)
            print(similarity)
            similarity = similarity[0,0]
        '''
        for i in range(total_doc):
            if self.length[i] == 0:
                similarity[i] = 0
            else:
                similarity[i] = similarity[i]/(self.length[i])/(query.length)
        return sorted(list(zip(range(total_doc), similarity)), key=lambda x: x[1], reverse=True)
    def rocchio_feedback(self, query, alpha, beta, pseudo_threshold = 10):
        adjust = sparse.dok_matrix((1, self.tf_idf.shape[1]))
        for result in self.cosine_similarity(query)[:pseudo_threshold]:
            adjust = adjust+self.tf_idf[result[0]]
        query.sparse = ((alpha * query.sparse.transpose()) + ((beta / pseudo_threshold) * adjust)).transpose()
    def rank(self, output, rocchio, lsi = True):
        print('ranking...')
        with open(output, 'w') as fp:
            for query in self.queries:
                if lsi:
                    query.sparse = query.sparse.transpose().dot(self.reduce_mapping).dot(self.sigma).transpose()
                if rocchio:
                    self.rocchio_feedback(query, 0.9, 0.2)
                for result in self.cosine_similarity(query)[:100]:
                    fp.write(query.number+' '+self.file_index[result[0]]+'\n')

    def parse(self, query_path, vocab_index):
        self.queries = Query.parse(query_path, vocab_index, self.index)
        print('parsing...')
