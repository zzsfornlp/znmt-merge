# some searching routines
from . import search, utils
from .search import State, SearchGraph
from collections import defaultdict

# extract n-best list from a search graph
def extract_nbest(sg, n, length_reward=0., normalizing_alpha=0., max_repeat_times=1):
    # length_reward could be accurate, but normalizing again brings approximity
    utils.zcheck(length_reward*normalizing_alpha==0., "It is meaningless to set them both!!")
    _sort_k = lambda one: (one.score_partial + length_reward*one.length) / (one.length ** normalizing_alpha)
    _TMP_KEY = "_extra_nbest"
    #
    cands = []
    stack_map = defaultdict(int)
    for one in sg.get_ends():
        _set_recusively_one(one, sg, n, _sort_k, stack_map, max_repeat_times)
        vs = one.get(_TMP_KEY)
        cands += vs
    if stack_map["RR"] > 0:
        utils.zlog("Repeat %s times!!" % stack_map["RR"], func="warn")
    v = sorted(cands, key=_sort_k, reverse=True)[:n]
    for one in v:
        one.mark_end()
    return v

def _set_recusively_one(one, sg, n, sort_k, stack_map, max_repeat_times):
    _TMP_KEY = "_extra_nbest"
    _PRUNE_KEY = "PRUNING_LIST"
    utils.zcheck(not sg.is_pruned(one), "Not scientific to call on pruned states!!")
    if one.get(_TMP_KEY) is None:
        if one.is_start():
            # rebuild without search-graph
            v = [State(sg=SearchGraph(origin_sg=one.sg))]
            one.set(_TMP_KEY, v)
        else:
            combined_list = one.get(_PRUNE_KEY)
            if combined_list is None or n<=1:
                combined_list = []
            utils.zcheck_ff_iter(combined_list, lambda x: x.get(_TMP_KEY) is None, "Not scientific pruning states!!")
            # sort them and get the largest ones
            # todo(warn): the real paths are already larger than combined paths, thus considering the largest real ones will be ok
            combined_list = sorted(combined_list, key=sort_k)[-(n-1):]
            combined_list.append(one)
            # recursive with their prevs
            cands = []
            for pp in reversed(combined_list):  # only to process one first
                # todo(warn): count number of links in the stack, maybe need better strategies
                link_sig = pp.sig_idlink()
                if stack_map[link_sig] >= max_repeat_times:
                    stack_map["RR"] += 1
                    # utils.zlog("Repeat once!!", func="warn")
                else:
                    stack_map[link_sig] += 1
                    _set_recusively_one(pp.prev, sg, n, sort_k, stack_map, max_repeat_times)
                    stack_map[link_sig] -= 1
                    for one_prev in pp.prev.get(_TMP_KEY):
                        new_state = State(prev=one_prev, action=pp.action)
                        pp.transfer_values(new_state)
                        cands.append(new_state)
            # and combining (could be more effective using HEAP or sth)
            v = sorted(cands, key=sort_k, reverse=True)[:n]
            one.set(_TMP_KEY, v)    # not on pruned states

# record
# todo(1. how many non-best actions, 2. expands (effective) rate and more analysis, 3. extract merging points)
class StateStater(object):
    def __init__(self):
        self.states = defaultdict(int)
        self.tags = defaultdict(int)
        self.num = 0
        self.step = 0

    def record(self, one_sg):
        them = one_sg.bfs()
        self.num += 1
        self.step += len(them)
        for s0, s1 in them:
            for one in s0+s1:
                self.states[one.state()] += 1
                for t in one.tags():
                    self.tags[t] += 1

    def report(self):
        if self.num > 0:
            utils.zlog("num=%s, steps=%s, steps/num=%.3f" % (self.num, self.step, (self.step+0.)/self.num), func="info")
            for ds in (self.states, self.tags):
                for k in sorted(list(ds.keys())):
                    utils.zlog("-> %s=%s(%.3f||%.3f)" % (k, ds[k], (ds[k]+0.)/self.num, (ds[k]+0.)/self.step), func="info")

# extract all the states from a sg
def extract_states(sg):
    HID_NAME = "hidv"
    target_dict = sg.target_dict
    src_inst = sg.src_info
    src_str = str(src_inst)
    states = []
    them = sg.bfs()
    for s0, s1 in them:
        for one in s0+s1:
            hid = one.get(HID_NAME)
            if hid is not None:
                pruner = one.get("PR_PRUNER")
                pruner_id = None if pruner is None else pruner.id
                state_value = {HID_NAME:hid.tolist(), "repr":str(one), "path":one.show_words(),
                               "id":one.id, "pruner":pruner_id, "state":one.state()}
                states.append(state_value)
    return [src_str, states]
