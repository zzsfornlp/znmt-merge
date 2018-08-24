# some extras

import numpy
import copy
try:
    import scipy.misc as misc
except:
    pass

# from Nematus
def hamming_distance_distribution(sentence_length, vocab_size, tau):
    #based on https://gist.github.com/norouzi/8c4d244922fa052fa8ec18d8af52d366
    c = numpy.zeros(sentence_length)
    for edit_dist in range(sentence_length):
        n_edits = misc.comb(sentence_length, edit_dist)
        #reweight
        c[edit_dist] = numpy.log(n_edits) + edit_dist * numpy.log(vocab_size)
        c[edit_dist] = c[edit_dist] - edit_dist / tau - edit_dist / tau * numpy.log(vocab_size)

    c = numpy.exp(c)
    c /= numpy.sum(c)
    return c

def edit_distance_distribution(sentence_length, vocab_size, tau):
    #from https://gist.github.com/norouzi/8c4d244922fa052fa8ec18d8af52d366
    c = numpy.zeros(sentence_length)
    for edit_dist in range(sentence_length):
        n_edits = 0
        for n_substitutes in range(min(sentence_length, edit_dist)+1):
            n_insert = edit_dist - n_substitutes
            current_edits = misc.comb(sentence_length, n_substitutes, exact=False) * \
                misc.comb(sentence_length+n_insert-n_substitutes, n_insert, exact=False)
            n_edits += current_edits
        c[edit_dist] = numpy.log(n_edits) + edit_dist * numpy.log(vocab_size)
        c[edit_dist] = c[edit_dist] - edit_dist / tau - edit_dist / tau * numpy.log(vocab_size)

    c = numpy.exp(c)
    c /= numpy.sum(c)
    return c

class RAML(object):
    @staticmethod
    def modify_data(insts, target_dict, opts):
        raml_tau = opts["raml_tau"]
        raml_samples = opts["raml_samples"]
        raml_reward = opts["raml_reward"]
        vocab_size = target_dict.get_wordceil()
        vocab = range(1, vocab_size)
        rets = []
        for one in insts:
            # must be (src, trg) pairs
            y_len = len(one.idxes[1])
            for i in range(raml_samples):
                tmp_one = copy.deepcopy(one)
                if raml_reward == "hd":
                    q = hamming_distance_distribution(y_len, vocab_size, raml_tau)
                    edits = numpy.random.choice(range(y_len), p=q)
                    if y_len > 1:
                        # choose random position except last
                        positions = numpy.random.choice(range(y_len - 1), size=edits, replace=False)
                    else:
                        positions = [0]
                    for position in positions:
                        tmp_one.idxes[1][position] = int(numpy.random.choice(vocab))
                elif raml_reward == "ed":
                    q = edit_distance_distribution(y_len, vocab_size, raml_tau)
                    edits = numpy.random.choice(range(y_len), p=q)
                    for e in range(edits):
                        if numpy.random.choice(["insertion", "substitution"]) == "insertion":
                            if y_len > 1:
                                # insert before last period/question mark
                                position = numpy.random.choice(range(y_len))
                            else:
                                position = 0
                            tmp_one.idxes[1].insert(position, numpy.random.choice(vocab))
                            y_len = len(tmp_one.idxes[1])
                        else:
                            if y_len > 1:
                                position = numpy.random.choice(range(y_len))
                                # only substitute if >1
                                if position == y_len - 1:
                                    #using choice of last position to mean deletion of random word instead
                                    del tmp_one.idxes[1][numpy.random.choice(range(y_len - 1))]
                                    y_len = len(tmp_one.idxes[1])
                                else:
                                    tmp_one.idxes[1][position] = numpy.random.choice(vocab)
                else:
                    raise NotImplementedError(raml_reward)
                rets.append(tmp_one)
        return rets

class ScheduledSamplerSetter(object):
    def __init__(self, opts):
        self.ss_start = opts["ss_start"]
        self.ss_scale = opts["ss_scale"]
        self.ss_min = opts["ss_min"]
        self.ss_k = opts["ss_k"]
        mode = opts["ss_mode"]
        if mode == "linear":
            self._ff = lambda x,k: 1.0-k*x
        elif mode == "exp":
            self._ff = lambda x,k: numpy.power(self.ss_k, x)
        elif mode == "isigm":
            self._ff = lambda x,k: k/(k+numpy.exp(x/k))
        elif mode == "none":
            self._ff = lambda x,k: 1.0
        else:
            raise NotImplementedError(mode)

    def transform_uidx(self, uidx):
        x = max(0, uidx-self.ss_start)
        return x/self.ss_scale

    def get_rate(self, uidx):
        x = self.transform_uidx(uidx)
        return max(self.ss_min, self._ff(x, self.ss_k))
