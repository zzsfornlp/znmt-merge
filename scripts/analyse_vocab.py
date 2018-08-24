#!/usr/bin/env python3

import json
import sys
from collections import OrderedDict, defaultdict

def main():
    word_wfreqs = defaultdict(int)
    word_cfreqs = defaultdict(int)
    count_words, count_sentences = 0, 0
    for line in sys.stdin:
        count_sentences += 1
        words_in = line.strip().split(' ')
        for w in words_in:
            count_words += 1
            word_wfreqs[w] += 1
        for w in set(words_in):
            word_cfreqs[w] += 1
    word_forms = list(word_wfreqs.keys())
    word_forms.sort(key=lambda x: word_wfreqs[x], reverse=True)
    worddict = OrderedDict()
    for ii, ww in enumerate(word_forms):
        worddict[ww] = (ii, word_wfreqs[ww], word_wfreqs[ww]/count_words, word_cfreqs[ww], word_cfreqs[ww]/count_sentences)
        print("|%-15s| %s" % (ww, worddict[ww]))
    # print(json.dumps(worddict, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
