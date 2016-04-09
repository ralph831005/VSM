import sys
import numpy as np
from collections import defaultdict
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
    print(eval(sys.argv[1], sys.argv[2]))
