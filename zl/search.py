# !! maybe the most important module, the searching/decoding part
# used for both testing and training

from collections import defaultdict, Iterable
import numpy as np
from . import utils, data
from .backends import BK

# the searching graph (tracking the relations between states)
class SearchGraph(object):
    def __init__(self, target_dict=None, src_info=None, origin_sg=None):
        self.target_dict = target_dict
        self.src_info = src_info
        self.ch_recs = defaultdict(list)
        self.root = None
        self.ends = []
        self.origin_sg = origin_sg

    #
    def get_real_self(self):
        if self.origin_sg is not None:
            return self.origin_sg
        else:
            return self

    def reg(self, state):
        if state.prev is not None:
            self.ch_recs[state.prev.id].append(state)
        else:
            self.root = state

    def childs(self, state):
        return self.ch_recs[state.id]

    def add_end(self, state):
        self.ends.append(state)

    def get_ends(self):
        return self.ends

    def is_pruned(self, one):
        return not one.is_end() and len(self.childs(one)) == 0

    def bfs(self):
        ret = []        # stayed or ended, pruned
        currents = [self.root]
        while len(currents) > 0:
            nexts = []
            ret.append(([], []))
            for one in currents:
                expands = self.childs(one)
                nexts += expands
                if not self.is_pruned(one):
                    ret[-1][0].append(one)
                else:
                    ret[-1][1].append(one)
                # sorting
                ret[-1][0].sort(key=lambda x: x.score_partial, reverse=True)
                ret[-1][1].sort(key=lambda x: x.score_partial, reverse=True)
            currents = nexts
        return ret

    def show_graph(self, td=None, print=True):
        if td is None:
            td = self.target_dict
        s = ""
        currents = [self.root]
        while len(currents) > 0:
            nexts = []
            currents.sort(key=lambda x: x.score_partial, reverse=True)
            for one in currents:
                head = "## id=%s|pid=%s|s=%.3f|%s|(%s)" % (one.id, one.pid, one.score_partial, one.state(), " ".join(data.Vocab.i2w(td, one.get_path("action_code"))))
                expands = self.childs(one)
                exp_strs = []
                for z in sorted(expands, key=lambda x: x.score_partial, reverse=True):
                    if not self.is_pruned(z):
                        nexts.append(z)
                    exp_strs.append("id=%s|w=%s|s=%.3f(%.3f)|%s" % (z.id, td.getw(z.action_code), z.action.score, np.exp(z.action.score), z.state()))
                head2 = "\n-> " + ", ".join(exp_strs) + "\n"
                s += head + head2
            s += "-----\n"
            currents = nexts
        if print:
            utils.zlog(s)
        return s

# the states in the searching graph (should be kept small)
class State(object):
    _state_id = 0
    @staticmethod
    def _get_id():
        State._state_id += 1
        return State._state_id
    @staticmethod
    def reset_id():
        State._state_id = 0

    def __init__(self, sg=None, action=None, prev=None, **kwargs):
        self.sg = sg
        self.action = action
        self.prev = prev        # previous state, if None then the start state
        self.length = 0         # length of the actions
        self.ended = False      # whether this state has been ended
        self.values = kwargs    # additional values & information
        self.caches = {}
        self._score_final = None
        self._score_partial = 0
        self._state = "NAN"     # nan, expand, end, pr*
        self._tags = []
        if prev is not None:
            self.length = prev.length + 1
            self._score_partial = action.score + prev._score_partial
            self.sg = prev.sg
        else:   # just put it as a padding
            self.action = Action(0, 0.)
        self.id = State._get_id()   # should be enough within python's int range
        if self.sg is not None:
            self.sg.reg(self)       # register in the search graph

    # for convenience, leave these out
    # @property: action, prev, id, length

    def __repr__(self):
        x = "ID=%s|PID=%s|LEN=%s|SS=%.3f(%.3f)|ACT=%s" % (self.id, self.pid, self.length, self._score_partial, self.score_final, self.action)
        td = self.sg.target_dict
        if td is not None:
            x += "|%s" % (td.getw(self.action_code),)
        return x

    def __str__(self):
        return self.__repr__()

    def show_words(self, td=None, print=False, verbose_level=0):
        if td is None:
            td = self.sg.target_dict
        s = []
        paths = self.get_path()
        for one in paths:
            s.append(one.descr_word(td, verbose_level))
        if print:
            utils.zlog(" ".join(s))
        return s

    def descr_word(self, td=None, verbose_level=0):
        if td is None:
            td = self.sg.target_dict
        if verbose_level==0:
            s = td.getw(self.action_code)
        elif verbose_level==1:
            s = "%s(%.3f)" % (td.getw(self.action_code), self.action.score)
        else:
            s = "%s(%s|%.3f)" % (td.getw(self.action_code), self.action, self.score_partial)
        return s

    @property
    def pid(self):
        if self.prev is None:
            return -1
        else:
            return self.prev.id

    @property
    def score_partial(self):
        return self._score_partial

    @property
    def score_final(self):
        return utils.Constants.MIN_V if self._score_final is None else self._score_final

    # for the actions
    @property
    def action_code(self):
        return int(self.action)

    def action_score(self, s=None):
        # todo: but will not update the states later, be careful about this
        if s is not None:
            # also change accumulate scores
            self._score_partial += (s-self.action.score)
            self.action.score = s
        return self.action.score

    # ---
    # each State only has one state
    def state(self, s=None):
        if s is not None:
            self._state = s
        return self._state

    # but could have multiple tags
    def tags(self, s=None):
        if s is not None:
            self._tags.append(s)
        return self._tags

    def set_score_final(self, s):
        utils.zcheck(self.ended, "Nonlegal final calculation for un-end states.")
        self._score_final = s

    def mark_end(self):
        self.state("END")   # also note state here
        self.ended = True
        if self.sg is not None:
            self.sg.add_end(self)

    def is_end(self):
        return self.ended

    # whether this is the starting state
    def is_start(self):
        return self.prev is None

    # about values
    def set(self, k, v):
        self.values[k] = v

    def add_list(self, k, v):
        if k not in self.values:
            self.values[k] = []
        utils.zcheck_type(self.values[k], list)
        self.values[k].append(v)

    def transfer_values(self, other, ruler=lambda x: str.islower(str(x)[0])):
        # todo(warn): specific rules, default start with lowercase
        r = {}
        for k in self.values:
            if ruler(k):
                r[k] = self.values[k]
        other.values = r

    def get(self, which=None):
        if which is None:
            return self
        elif self.values and which in self.values:
            return self.values[which]
        elif hasattr(self, which):
            return getattr(self, which)
        else:
            return None

    def get_path(self, which=None, maxlen=-1):
        # todo(warn): negative trick
        if maxlen == 0:
            return []
        if self.prev is None:
            l = []
        else:
            l = self.prev.get_path(which, maxlen-1)
            v = self.get(which)
            l.append(v)
        return l

    # signatures
    def sig_ngram_list(self, n):
        if n not in self.values:
            # calculate it on the fly
            if self.is_start():
                self.values[n] = [""]*n
            else:
                pp = self.prev.sig_ngram_list(n)
                self.values[n] = [str(self.action_code)] + pp[:-1]
        return self.values[n]

    # padded version of list, used for ngram-decoder
    def sig_ngram_tlist(self, n, pad):
        v = -n     # todo(warn) ...
        if v not in self.values:
            # calculate it on the fly
            if self.is_start():
                self.values[v] = [pad]*n
            else:
                pp = self.prev.sig_ngram_tlist(n, pad)
                self.values[v] = [self.action_code] + pp[:-1]
        return self.values[v]

    def sig_ngram(self, n):
        if n not in self.caches:
            ll = self.sig_ngram_list(n)
            self.caches[n] = "|".join(ll)
        return self.caches[n]

    def sig_idlink(self):
        if "_il" not in self.caches:
            self.caches["_il"] = "%s|%s" % (self.id, self.pid)
        return self.caches["_il"]

    # special: use specific names "hidv", "CC"
    def get_hidv(self):
        if "hidv" not in self.values:
            cc, idx = self.values["CC"]
            slice = BK.pick_batch_elem(cc['hid'][0]['H'], idx)
            self.values["hidv"] = BK.get_value_np(slice)
        return self.values["hidv"]

# action
class Action(object):
    def __init__(self, action_code, score):
        self.action_code = action_code
        self.score = score
        # self.values = kwargs

    def __repr__(self):
        return "[%s/%.3f/%.3f]" % (self.action_code, self.score, np.exp(self.score))

    def __str__(self):
        return self.__repr__()

    def __int__(self):
        return self.action_code

# state + actions (to be scored, not fully expanded)
class StateCands(object):
    pass

# =========================
# the searcher
class Searcher(object):
    pass

# the loss builder
class Losser(object):
    pass
