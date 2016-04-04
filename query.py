import numpy as np
import os
import math
from collections import defaultdict
total_doc = 46972
total_grams = 1193467
start = { 'topic': '<topic>', 'number': '<number>', 'title': '<title>', 'question': '<question>', 'narrative': '<narrative>', 'concepts': '<concepts>' }
close = { 'topic': '</topic>', 'number': '</number>', 'title': '</title>', 'question': '</question>', 'narrative': '</narrative>', 'concepts': '</concepts>' }
seperators = '、，。'
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
                grams = defaultdict(lambda : defaultdict(int))
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

