# data iterators and dictionaries

from . import utils
import json, os, sys
import numpy as np
from collections import Iterable

# ========== Vocab ========== #
# The class of vocabulary or dictionary: conventions: 0:zeros, 1->wordceil:words, then:special tokens
class Vocab(object):
    SPECIAL_TOKENS = ["<unk>", "<bos>", "<eos>", "<pad>", "<err>"]    # PLUS <non>: 0
    NON_TOKEN = "<non>"

    @staticmethod
    def _build_vocab(stream, rthres, fthres, specials):
        word_freqs = {}
        # read
        for w in stream:
            if w not in word_freqs:
                word_freqs[w] = 0
            word_freqs[w] += 1
        # sort
        words = [w for w in word_freqs]
        words = sorted(words, key=lambda x: (word_freqs[x], x), reverse=True)
        # write with filters
        v = {Vocab.NON_TOKEN: 0}    # todo(warn) hard-coded
        cur_idx = 1
        for ii, ww in enumerate(words):
            rank, freq = ii, word_freqs[ww]
            if rank <= rthres and freq >= fthres:
                v[ww] = cur_idx
                cur_idx += 1
        # add specials
        for one in specials:
            v[one] = len(v)
        utils.zlog("Build Dictionary: (origin=%s, final=%s, special=%s as %s)." % (len(words), len(v), len(specials)+1, specials))
        return v, words

    def __init__(self, d=None, s=None, fname=None, rthres=100000, fthres=1, specials=None):
        # three possible sources: d=direct-dict, s=iter(str), f=tokenized-file
        if specials is None:    # todo(warn) default is like this
            specials = Vocab.SPECIAL_TOKENS
        self.d = {}
        if d is not None:
            self.d = d
            # insure specials are in here
            utils.zcheck_ff_iter(specials, lambda x: x in self.d, "not included specials", "warn", _forced=True)
            utils.zcheck_ff(self.d[Vocab.NON_TOKEN], lambda x: x==0, "unequal to 0", "warn", _forced=True)
        elif s is not None:
            with utils.Timer(tag="vocab", info="build vocabs from stream."):
                self.d, _ = Vocab._build_vocab(s, rthres, fthres, specials)
        elif fname is not None:
            with utils.Timer(tag="vocab", info="build vocabs from corpus %s" % fname):
                with utils.zopen(fname) as f:
                    self.d, _ = Vocab._build_vocab(utils.Helper.stream_on_file(f), rthres, fthres, specials)
        else:
            utils.zfatal("No way to init Vocab.")
        # reverse vocab
        self.v = ["" for _ in range(len(self.d))]
        for k in self.d:
            self.v[self.d[k]] = k

    def write(self, wf):
        with utils.zopen(wf, 'w') as f:
            f.write(json.dumps(self.d, ensure_ascii=False, indent=2))
            utils.zlog("-- Write Dictionary to %s: Finish %s." % (wf, len(self.d)), func="io")

    @staticmethod
    def read(rf):
        with utils.zopen(rf) as f:
            df = json.loads(f.read())
            utils.zlog("-- Read Dictionary from %s: Finish %s." % (rf, len(df)), func="io")
            return Vocab(d=df)

    # queries
    def get_wordceil(self):
        return self.d["<unk>"]+1      # excluding special tokens except unk

    # special tokens
    def get_ending_tokens(self):
        rr = [self.eos]
        if '.' in self.d:
            rr.append(self.d['.'])
        return rr

    @property
    def non(self):
        return 0

    @property
    def eos(self):
        return self.d["<eos>"]

    @property
    def pad(self):
        return self.d["<pad>"]

    @property
    def unk(self):
        return self.d["<unk>"]

    @property
    def bos(self):
        return self.d["<bos>"]

    @property
    def err(self):
        return self.d["<err>"]

    # getting them
    def getw(self, index):
        return self.v[index]

    def __getitem__(self, item):
        utils.zcheck_type(item, str)
        if item in self.d:
            return self.d[item]
        else:
            return self.d["<unk>"]

    def __len__(self):
        return len(self.d)

    # words <=> indexes (be aware of lists)
    @staticmethod
    def w2i(dicts, ss, add_eos=True, use_factor=False, factor_split='|'):
        # Usage: list(Vocab), list(str) => list(list(int))[use_factor] / list(int)[else]
        if not isinstance(dicts, list):
            dicts = [dicts]
        utils.zcheck_ff_iter(dicts, lambda x: isinstance(x, Vocab), "Wrong type")
        # lookup
        tmp = []
        for w in ss:
            if use_factor:
                idx = [dicts[i][f] for (i,f) in enumerate(w.split(factor_split))]
            else:
                idx = dicts[0][w]
            tmp.append(idx)
        if add_eos:
            tmp.append([d.eos for d in dicts] if use_factor else dicts[0].eos)  # add eos
        return tmp

    @staticmethod
    def i2w(dicts, ii, rm_eos=True, factor_split='|'):
        # Usage: list(Vocab), list(int)/list(list(int)) => list(str)
        if not isinstance(dicts, list):
            dicts = [dicts]
        utils.zcheck_ff_iter(dicts, lambda x: isinstance(x, Vocab), "Wrong type")
        tmp = []
        # get real list
        real_ii = ii
        if len(ii)>0 and rm_eos and ii[-1]==dicts[0].eos:
            real_ii = ii[:-1]
        # transform each token
        for one in real_ii:
            if not isinstance(one, list):
                one = [one]
            utils.zcheck_matched_length(one, dicts)
            tmp.append(factor_split.join([v.getw(idx) for v, idx in zip(dicts, one)]))
        return tmp

# ========== Instance and Reader ========== #
# handling the batching of instances, also filtering, sorting, recording, etc.
class BatchArranger(object):
    def __init__(self, streamer, batch_size, maxibatch_size, outliers, single_outlier, sorting_keyer, tracking_order, shuffling):
        self.streamer = streamer
        self.outliers = [] if outliers is None else outliers
        self.single_outliers = (lambda x: False) if single_outlier is None else single_outlier
        self.sorting_keyer = sorting_keyer  # default(None) no sorting
        self.tracking_order = tracking_order
        self.tracking_list = None
        self.batch_size = batch_size
        # todo(notice): if <=0 then read all at one time and possibly sort all
        self.maxibatch_size = maxibatch_size if maxibatch_size>0 else utils.Constants.MAX_V
        self.shuffling = shuffling  # shuffle inside the maxi-batches?

    def __len__(self):
        return sum([len(i) for i in self.arrange_batches()])

    @property
    def k(self):
        # return self.sort_ksize
        return self.batch_size * self.maxibatch_size

    # getter and checker for batch_size (might have unknown effects if changed in the middle)
    def bsize(self, bs=None):
        if bs is None:
            return self.batch_size
        else:
            self.batch_size = int(bs)
            return self.batch_size

    # the iterating generator
    def arrange_batches(self):
        # read streams of batches of items, return list-of-items
        buffer = []
        if self.tracking_order:
            self.tracking_list = []
        inst_stream = self.streamer.stream()
        while True:
            # read into buffer
            for idx, one in enumerate(inst_stream):
                # filters out (like short or long instances)
                filtered_flag = False
                for tmp_filter in self.outliers:
                    if tmp_filter(one):
                        filtered_flag = True
                        break
                if filtered_flag:
                    continue
                # need to generate single instance
                if self.single_outliers(one):
                    if self.tracking_order:
                        self.tracking_list.append(idx)
                    else:
                        utils.zfatal("Should not have outliers when training!!")
                    yield [one]
                else:
                    # adding in buffer with index
                    buffer.append((idx, one))
                    if len(buffer) == self.k:  # must reach equal because +=1 each time
                        break
            # time for yielding
            if len(buffer) > 0:
                # sorting
                sorted_buffer = buffer
                if self.sorting_keyer is not None:
                    sorted_buffer = sorted(buffer, key=(lambda x: self.sorting_keyer(x[1])))
                # prepare buckets
                buckets = [sorted_buffer[_s:_s+self.batch_size] for _s in range(0, len(sorted_buffer), self.batch_size)]
                # another shuffle?
                if self.shuffling:
                    utils.Random.shuffle(buckets, "data_bucket")
                # final yielding
                for oneb in buckets:
                    if self.tracking_order:
                        self.tracking_list += [_one[0] for _one in oneb]
                    yield [_one[1] for _one in oneb]
                buffer = []
            else:
                break

    def get_tracking_list(self):
        return self.tracking_list

    # def restore_order(self, x):
    #     # rearrange back according to self.tracking_list (caused by sorting and shuffling)
    #     if not self.tracking_order:
    #         return x
    #     # utils.zcheck(self.tracking_order, "Tracking function has not been opened", _forced=True)
    #     utils.zcheck_matched_length(self.tracking_list, x, _forced=True)
    #     ret = [None for _ in x]
    #     for idx, one in zip(self.tracking_list, x):
    #         utils.zcheck_type(ret[idx], type(None), "Wrong tracking list, internal error!!", _forced=True)
    #         ret[idx] = one
    #     return ret

# single instance, batched-version should be handled when modeling since it will be quite different for different models
class Instance(object):
    pass

# read and return Instance
class InstanceReader(object):
    def stream(self):
        raise NotImplementedError()

# ========== Specified Instance and DataIter for simple seq2seq texts ========== #
class TextInstance(Instance):
    # multiple sequences of indexes
    def __init__(self, words, idxes):
        utils.zcheck_matched_length(words, idxes)
        self.words = words
        self.idxes = idxes

    def __repr__(self):
        return "||".join([" ".join(one) if isinstance(one[0],str) else str(len(one)) for one in self.words])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        # how many instances
        return len(self.idxes)

    def __getitem__(self, item):
        return self.idxes[item]

    def get_words(self, i):
        return self.words[i]

    def get_lens(self):
        return [len(ww) for ww in self.words]

    # for evaluating and analysing
    def set(self, k, v):
        if "values" not in self.__dict__:
            setattr(self, "values", {})
        self.values[k] = v

    def get(self, k):
        if k not in self.values:
            return None
        return self.values[k]

    def extract(self):
        ret = []
        for i in range(len(self)):
            # todo(warn): bad discrimination criteria
            if isinstance(self.idxes[i][0], Iterable):
                ret.append(TextInstance(self.words[i], self.idxes[i]))
            else:   # just one
                ret.append(TextInstance([self.words[i]], [self.idxes[i]]))
        return ret

class TextInstanceRangeOutlier(object):
    # detect outlier, return True if outlier detected
    def __init__(self, a=utils.Constants.MIN_V, b=utils.Constants.MAX_V):
        self.a = a
        self.b = b

    def __call__(self, inst):  # True if any not in [a, b)
        return any(not (x >= self.a and x < self.b) for x in inst.get_lens())

class TextInstanceLengthSorter(object):
    # information about sort-by-lengths
    def __init__(self, prior):
        self.prior = prior

    def __call__(self, inst):
        # return the key for one inst
        MAXLEN, UNIT = 1000, 4  # todo(warn), magic number
        acc = 0
        lens = inst.get_lens()
        for i in reversed(self.prior):
            acc = acc * MAXLEN + lens[i] // UNIT
        return acc

# read from files
class TextFileReader(InstanceReader):
    def __init__(self, files, vocabs, multis, shuffling, shuffling0):
        utils.zcheck_matched_length(files, vocabs, _forced=True)
        if not isinstance(multis, Iterable):
            utils.zcheck_type(multis, bool, _forced=True)
            multis = [multis for _ in range(len(files))]
        else:
            utils.zcheck_matched_length(files, multis, _forced=True)
        self.files = files
        self.vocabs = vocabs
        self.multis = multis
        self.shuffling = shuffling
        self.num_insts = -1
        # weather reading multi lines
        if shuffling and any(multis):
            utils.zfatal("Not implemented for shuffling multi-line files.")
        self._readers = [TextFileReader._read_one if not m else TextFileReader._read_one_multi for m in multis]
        #
        if shuffling0:
            with utils.Timer(tag="shuffle", info="shuffle the file corpus before training."):
                self.files = TextFileReader.shuffle_corpus(self.files, openfd=False)

    def __len__(self):
        # return num of instances (use cached value)
        if self.num_insts < 0:
            self.num_insts = sum([1 for _ in self.stream()])
        return self.num_insts

    @staticmethod
    def shuffle_corpus(files, openfd=True):
        # global shuffle, creating tmp files on current dir
        with utils.zopen(files[0]) as f:
            lines = [[i.strip()] for i in f]
        for ff in files[1:]:
            with utils.zopen(ff) as f:
                for i, li in enumerate(f):
                    lines[i].append(li.strip())
        utils.Random.shuffle(lines, "corpus")
        # write
        filenames_shuf = []
        for ii, ff in enumerate(files):
            path, filename = os.path.split(os.path.realpath(ff))
            filenames_shuf.append(filename+'.%d.shuf'%ii)   # avoid identical file-names
            with utils.zopen(filenames_shuf[-1], 'w') as f:
                for l in lines:
                    f.write(l[ii]+"\n")
        if openfd:
            # read
            fds = [utils.zopen(_f) for _f in filenames_shuf]
            return fds
        else:
            return filenames_shuf

    @staticmethod
    def _read_one(fd, vv):
        line = fd.readline()
        if len(line) <= 0:
            return None
        else:
            words = line.strip().split()
            idxes = Vocab.w2i(vv, words, add_eos=True, use_factor=False)
            return words, idxes

    @staticmethod
    def _read_one_multi(fd, vv):
        # separated by multiple "\n"
        words = None
        while True:
            line = fd.readline()
            if len(line) <= 0:
                return None
            words = line.strip().split()
            if len(words) > 0:
                break
        rets = ([], [])
        while len(words) > 0:
            rets[0].append(words)
            rets[1].append(Vocab.w2i(vv, words, add_eos=True, use_factor=False))
            line = fd.readline()
            words = line.strip().split()
            if len(line) <= 0 or len(words) <= 0:
                break
        return rets

    def stream(self):
        if self.shuffling:
            # todo(warn) only shuffling when training
            with utils.Timer(tag="shuffle", info="shuffle the file corpus"):
                fds = TextFileReader.shuffle_corpus(self.files)
        else:
            fds = [utils.zopen(one) for one in self.files]
        # read them and yield --- checking length
        idx = 0
        while True:
            insts = [self._readers[i](fds[i], self.vocabs[i]) for i in range(len(self.files))]
            if any([_x is None for _x in insts]) and any([_x is not None for _x in insts]):
                utils.zfatal("EOF unmatched for %s." % insts)
            if insts[0] is None:
                break
            words = [_x[0] for _x in insts]
            idxes = [_x[1] for _x in insts]
            idx += 1
            # todo(warn): DEBUG
            # print(words)
            # if len(words[1])<10:
            #     print("oh no")
            self.num_insts = max(self.num_insts, idx)
            yield TextInstance(words, idxes)
        # close
        for ffd in fds:
            ffd.close()

# one call for convenience
def get_arranger(files, vocabs, multis, shuffling_corpus, shuflling_buckets, sort_prior, batch_size, maxibatch_size, max_len, min_len, one_len, shuffling0):
    streamer = TextFileReader(files, vocabs, multis, shuffling_corpus, shuffling0)
    tracking_order = True if maxibatch_size<=0 else False   # todo(warn): -1 for dev/test
    arranger = BatchArranger(streamer=streamer, batch_size=batch_size, maxibatch_size=maxibatch_size, outliers=[TextInstanceRangeOutlier(min_len, max_len)], single_outlier=TextInstanceRangeOutlier(min_len, one_len), sorting_keyer=TextInstanceLengthSorter(sort_prior), tracking_order=tracking_order,shuffling=shuflling_buckets)
    return arranger

def get_arranger_simple(files, vocabs, multis, batch_size):
    streamer = TextFileReader(files, vocabs, multis, False, False)
    tracking_order = False
    arranger = BatchArranger(streamer=streamer, batch_size=batch_size, maxibatch_size=1, outliers=None, single_outlier=None, sorting_keyer=None, tracking_order=tracking_order,shuffling=False)
    return arranger
