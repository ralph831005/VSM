#!/usr/bin/python3
import pickle
import numpy as np
import os
import math
from collections import defaultdict
from query_sparse import Query
from scipy import sparse
import scipy.sparse.linalg as la
import glob
total_doc = 46972
total_grams = 1193467
def save_obj(obj, name):
    with open(name, 'wb') as fp:
        pickle.dump(obj, fp, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open(name, 'rb') as fp:
        return pickle.load(fp)
class VSM:
    def __init__(self, invert_file_path, file_list, file_directory):
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
        tmp_file_length = dict()
        out_file_length = 'file_length.bi'
        if os.path.exists(out_file_length):
            self.file_length = load_obj(out_file_length)
        else:
            for f in glob.glob(file_directory+'*/*/*'):
                with open(f) as fp:
                    lines = ''.join([x.strip() for x in fp.readlines()])
                    title = lines.split('<title>')[1].split('</title>')[0].strip()
                    p = ''.join([x.split('</p>')[0] for x in lines.split('<p>')[1:]])
                    tmp_file_length[f.split('/')[-1].lower()] = len(p) + len(title)
            self.file_length = [tmp_file_length[x] for x in self.file_index]
            save_obj(self.file_length, out_file_length)
        self.avg_length = np.mean(self.file_length)
    def weight(self, cache = True, tf_func=lambda tf, k, b, d: (tf * (k+1.0))/(tf + k*(1-b+b*d)), idf_func=lambda N, nq: math.log((N - nq + 0.5)/(nq+0.5))):
        out_data = '/tmp3/ralph831005/IR/coo_weight.npy'
        out_index = '/tmp3/ralph831005/IR/index.binary'
        out_length = '/tmp3/ralph831005/IR/length.npy'
        print('weighting start')
        if cache and os.path.exists(out_data):
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
                self.idf[tmp] = idf_func(total_doc, float(bigram[2]))
                lines = [[int(x) for x in inverted_file.readline().strip().split()] for i in range(int(bigram[2]))]
                max_tf = max(list(zip(*lines))[1])
                for doc in lines:
                    data.append((doc[0], tmp, tf_func(doc[1], 1.4, 0.75, float(self.file_length[doc[0]])/self.avg_length) * self.idf[tmp]))
                    self.length[doc[0]] += data[-1][-1]**2
                counter += 1
        for i in range(len(self.length)):
            self.length[i] = (self.length[i]**(0.5))
        data = np.transpose(data)
        self.tf_idf = sparse.csr_matrix((data[2], (data[0], data[1])), shape=(total_doc, total_grams))
        if cache:
            np.save(out_length, self.length)
            save_obj(self.index, out_index)
            save_obj(self.tf_idf, out_data)
        print('weighting done')
    def lsi(self, n_sigular_value=200):
        print('lsi start')
        self.tf_idf, sigma, v = la.svds(self.tf_idf, k = n_sigular_value)
        self.reduce_mapping = v.transpose()
        self.sigma = la.inv(sparse.csr_matrix((sigma, (range(n_sigular_value), range(n_sigular_value))), shape=(n_sigular_value, n_sigular_value))).toarray()
        print('lsi done')
    def cosine_similarity(self, query):
        similarity = np.transpose(sparse.csr_matrix(self.tf_idf.dot(query.sparse)).toarray())[0]
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
    def rank(self, output, rocchio, lsi = True, alpha=0.9, beta=0.2, pseudo_threshold=10):
        print('ranking...')
        with open(output, 'w') as fp:
            for query in self.queries:
                if lsi:
                    query.sparse = query.sparse.transpose().dot(self.reduce_mapping).dot(self.sigma).transpose()
                if rocchio:
                    self.rocchio_feedback(query, alpha, beta, pseudo_threshold)
                for result in self.cosine_similarity(query)[:100]:
                    fp.write(query.number+' '+self.file_index[result[0]]+'\n')

    def parse(self, query_path, vocab_index):
        self.queries = Query.parse(query_path, vocab_index, self.index)
        print('parsing...')
