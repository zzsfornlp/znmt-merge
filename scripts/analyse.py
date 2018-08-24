#!/bin/python3

# analyse the corpus

import sys
from collections import defaultdict
def main():
    num_words = num_sents = 0
    dl = defaultdict(int)   # len -> lfreq
    fw = defaultdict(int)   # word -> wfreq
    ff = defaultdict(int)   # wfreq -> wwfreq
    # collect info
    for l in sys.stdin:
        fs = l.split()
        num_words += len(fs)
        num_sents += 1 if len(fs)>0 else 0
        dl[len(fs)] += 1
        for w in fs:
            fw[w] += 1
    for w in fw:
        ff[fw[w]] += 1
    # print info
    print("==> Summary info: %s(sents), %s(words), %s(w/s), %s(diff-words)" % (num_sents, num_words, num_words/num_sents, len(fw)))
    print("==> Length info")
    lens = sorted(list(dl.keys()))
    cum = 0
    for k in lens:
        cum += dl[k]
        print("%s: %s (cum:%s)" % (k, dl[k], cum))
    print("==> Words info")
    cur = 0
    max_freq = max(ff)
    for i in reversed(range(max_freq)):
        if i in ff:
            cur += ff[i]
            print("%s: %s(+%s)" % (i, cur, ff[i]))

if __name__ == '__main__':
    main()
