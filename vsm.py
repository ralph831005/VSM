#!/usr/bin/python3
import numpy as np
import os
import math
from collections import defaultdict
invert_file_path = '/tmp3/ralph831005/IR/model/inverted-file'
total_doc = 46972
total_grams = 1193467
start = { 'topic': '<topic>', 'number': '<number>', 'title': '<title>', 'question': '<question>', 'narrative': '<narrative>', 'concepts': '<concepts>' }
close = { 'topic': '</topic>', 'number': '</number>', 'title': '</title>', 'question': '</question>', 'narrative': '</narrative>', 'concepts': '</concepts>' }
seperators = '、，。'
class VSM:
    def __init__(self):
        self.tf_idf = [dict() for i in range(total_doc)]
        self.v_length = np.zeros(total_doc)
        self.idf = np.zeros(total_grams)
        self.index = defaultdict(dict)
    def weight(self, file_path=invert_file_path, tf_func=lambda tf, mtf: 0.5 * (float(tf)/float(mtf))):
        with open(file_path, 'r') as inverted_file:
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
        print('weighting done')
        
    def consine_similarity(self, query):
        similarity = np.zeros(total_doc)
        for i, doc in enumerate(self.tf_idf):
            for k, v in query.sparse.items():
                if k in doc:
                    similarity[i] += doc[k]*v
            similarity[i] = similarity[i]*(self.v_length[i]**(0.5))
        return sorted(list(zip(range(total_doc), similarity)), key=lambda x: x[1], reverse=True)

class Query:
    def __init__(self, number, grams, index):
        self.number = number
        self.sparse = dict()
        for i, v in grams.items():
            if i in index:
                for j, term in v.items():
                    if j in index[i]:
                        self.sparse[index[i][j]] = term
    def extract(token, vocab_index, grams):
        #unigram
        indexed = []
        for ch in token:
            if ch in vocab_index:
                indexed.append(vocab_index[ch])
                grams[vocab_index[ch]][-1] += 1
        for bi in list(zip(indexed, indexed[1:])):
            grams[bi[0]][bi[1]] += 1

    def parse(file_path, vocab_index, index):
        queries = []
        with open(file_path) as fp:
            content = ''.join([x.strip() for x in fp.readlines()])
            for unparsed_query in content.split(start['topic'])[1:]:
                grams = dict(defaultdict(int))
                number = unparsed_query.split(start['number'])[1].split(close['number'])[0][-3:]
                title = unparsed_query.split(start['title'])[1].split(close['title'])[0].split(seperators)
                question = unparsed_query.split(start['question'])[1].split(close['question'])[0][2:].split(seperators)
                narrative = unparsed_query.split(start['narrative'])[1].split(close['narrative'])[0][6:].split(seperators)
                concepts = unparsed_query.split(start['concepts'])[1].split(close['concepts'])[0].split(seperators)
                for token in title:
                    Query.extract(token, vocab_index, grams)
                for token in question:
                    Query.extract(token, vocab_index, grams)
                for token in narrative:
                    Query.extract(token, vocab_index, grams)
                for token in concepts:
                    Query.extract(token, vocab_index, grams)
                
                queries.append(Query(number, grams, index))
        return queries



if __name__ == '__main__':
    vocab_index = dict()
    with open('/tmp3/ralph831005/IR/model/vocab.all') as fp:
        for i, vocab in enumerate(fp):
            vocab_index[vocab.strip()] = i

    vsm = VSM()
    vsm.weight(file_path='test.in', tf_func = lambda tf, mtf: float(tf)/float(mtf))
    queries = Query.parse('/tmp3/ralph831005/IR/query/query-train.xml', vocab_index, vsm.index)
    print(queries[0].sparse)
