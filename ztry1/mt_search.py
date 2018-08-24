# first write a full process for standard beam-decoding, then use it as a template to see how to re-factor

# the main searching processes
from zl.search import State, Action, SearchGraph
from zl.search2 import extract_nbest
from zl.model import Model
from zl import utils, data
from . import mt_layers as layers
import numpy as np
from collections import defaultdict
from queue import PriorityQueue
from .mt_par import CovChecker

##
from zl.backends.common import ResultTopk
IDX_DIM = ResultTopk.IDX_DIM
VAL_DIM = ResultTopk.VAL_DIM

MTState = State
MTAction = Action

def search_init():
    State.reset_id()

# a padding version of greedy search
# todo(warn) normers and pruners are not used for this greedy decoding (mostly used for dev)
def search_greedy(models, insts, target_dict, opts, normer, sstater, para_extractor):
    xs = [i[0] for i in insts]
    xwords = [i.get_words(0) for i in insts]
    Model.new_graph()
    for _m in models:
        _m.refresh(False)
    _m0 = models[0]     # maybe common for all models, todo(warn) this inhibit certain varieties of model diff
    bsize, finish_size = len(xs), 0
    opens = [State(sg=SearchGraph()) for _ in range(bsize)]
    ends = [None for _ in range(bsize)]
    yprev = [-1 for _ in range(bsize)]
    decode_maxlen = opts["decode_len"]
    decode_maxratio = opts["decode_ratio"]
    # if need to get att-weights (todo(warn))
    need_att = opts["decode_replace_unk"]
    caches = []
    eos_id = target_dict.eos
    for step in range(decode_maxlen):
        one_cache = []
        if step==0:
            for mi, _m in enumerate(models):
                cc = _m.start(xs)
                one_cache.append(cc)
        else:
            for mi, _m in enumerate(models):
                cc = _m.step(caches[-1][mi], yprev, utils.Helper.stream_rec(opens))
                one_cache.append(cc)
        caches.append(one_cache)
        # prepare next steps
        results = layers.BK.average([one["results"] for one in one_cache])
        results_topk = layers.BK.topk(results, 2)       # todo(warn): to avoid err states
        if need_att:
            atts_e = layers.BK.average([one["att"] for one in one_cache])
            atts_v = layers.BK.get_value_np(atts_e)
        else:
            atts_v = [None for _ in range(bsize)]
        for j in range(bsize):
            cur_xsrc = xwords[j] if need_att else None
            rr0 = results_topk[j]
            rr = _m0.explain_result_topkp(rr0)
            one_attv = atts_v[j]
            if ends[j] is None:
                # cur off if one of the criteria says it is time to end.
                # todo(warn): upper length pruner
                force_end = ((step+1) >= min(decode_maxlen, len(xs[j])*decode_maxratio))
                if not force_end:
                    next_y, next_score = rr[0][IDX_DIM], rr[0][VAL_DIM]
                else:
                    # todo(warn) ignore eos score for force_end
                    next_y, next_score = eos_id, 0.
                # check eos
                if next_y == eos_id:
                    ends[j] = State(prev=opens[j], action=Action(next_y, next_score), attw=one_attv, attention_src=cur_xsrc)
                    ends[j].mark_end()
                    finish_size += 1
                else:
                    opens[j] = State(prev=opens[j], action=Action(next_y, next_score), attw=one_attv, attention_src=cur_xsrc)
            else:
                next_y = 0
            yprev[j] = next_y
        if finish_size >= bsize:
            break
    # return them
    ret = [[s] for s in ends]
    for ranked_beam in ret:
        para_extractor.record(ranked_beam)
    return ret

# a padding version of sampling
# todo(warn) normers and local pruners are utilized
def search_sample(models, insts, target_dict, opts, normer, sstater, para_extractor):
    xs = [i[0] for i in insts]
    Model.new_graph()
    for _m in models:
        _m.refresh(False)
    _m0 = models[0]     # maybe common for all models, todo(warn) this inhibit certain varieties of model diff
    bsize, finish_size = len(xs), 0
    # pruners
    pr_local_expand = opts["pr_local_expand"]
    pr_local_diff = opts["pr_local_diff"]
    pr_local_penalty = opts["pr_local_penalty"]
    sample_size = opts["beam_size"]
    decode_maxlen = opts["decode_len"]
    decode_maxratio = opts["decode_ratio"]
    # init them (padded)
    opens, ends, yprev = [], [], []
    for i in range(bsize):
        sg0 = SearchGraph()
        opens.append([])
        ends.append([])
        for j in range(sample_size):
            opens[i].append(State(sg=sg0))
            ends[i].append(None)
            yprev.append(-1)
    # start to go
    caches = []
    eos_id = target_dict.eos
    for step in range(decode_maxlen):
        one_cache = []
        if step==0:
            for mi, _m in enumerate(models):
                cc = _m.start(xs, repeat_time=sample_size)
                one_cache.append(cc)
            # todo: somewhat repeated here
            _tmp_insts = utils.Helper.repeat_list(insts, sample_size)
            _tmp_pred_lens = np.average([_m.predict_length(_tmp_insts, cc=_c) for _m, _c in zip(models, one_cache)], axis=0)
            pred_lens = utils.Helper.shrink_list(_tmp_pred_lens, sample_size)
        else:
            for mi, _m in enumerate(models):
                cc = _m.step(caches[-1][mi], yprev, utils.Helper.stream_rec(opens))
                one_cache.append(cc)
        caches.append(one_cache)
        # prepare next steps
        results = layers.BK.average([one["results"] for one in one_cache])
        results_topk = layers.BK.topk(results, pr_local_expand)
        for i in range(bsize):
            force_end = ((step+1) >= min(decode_maxlen, len(xs[i])*decode_maxratio))
            for j in range(sample_size):
                cur_idx = i*sample_size+j
                rr0 = results_topk[cur_idx]
                rr = _m0.explain_result_topkp(rr0)
                if ends[i][j] is None:
                    # select
                    if force_end:
                        # todo(warn) ignore eos score for force_end
                        next_y, next_score = eos_id, 0.
                    else:
                        probs, rr_final = Pruner.local_prune_score(rr, pr_local_expand, pr_local_diff, pr_local_penalty)
                        selected = utils.Random.multinomial_select(probs, "sample")
                        next_y = rr_final[selected][IDX_DIM]
                        next_score = rr_final[selected][VAL_DIM]
                    # assign
                    # check eos
                    if next_y == eos_id:
                        ends[i][j] = State(prev=opens[i][j], action=Action(next_y, next_score))
                        ends[i][j].mark_end()
                        finish_size += 1
                    else:
                        opens[i][j] = State(prev=opens[i][j], action=Action(next_y, next_score))
                else:
                    next_y = 0
                yprev[cur_idx] = next_y
        if finish_size >= bsize*sample_size:
            break
    # final normer
    normer(ends, pred_lens)
    final_list = [sorted(beam, key=lambda x: x.score_final, reverse=True) for beam in ends]
    for ranked_beam in final_list:
        para_extractor.record(ranked_beam)
    return final_list

class BatchedHelper(object):
    def __init__(self, opens):
        # opens is list of list
        self.bsize = None
        self.shapes = None
        self.bases = None
        self._build(opens)

    def _build(self, opens):
        b = [0]
        for ss in opens:
            b.append(b[-1]+len(ss))
        self.bsize = len(opens)
        self.shapes = [len(one) for one in opens]
        self.bases = b

    def get_basis(self, i):
        return self.bases[i]

    def rerange(self, nexts):
        # nexts is list of list of (prev-idx, next_token), return orders, next_ys
        utils.zcheck_matched_length(nexts, self.shapes)
        bv_orders = []
        new_ys = []
        break_bv, break_bi = False, False
        for i in range(len(nexts)):
            nns = nexts[i]
            bas = self.bases[i]
            if len(nns) != self.shapes[i]:
                break_bi = True
            for one in nns:
                j = one[0]
                if (len(bv_orders)==0 and bas+j!=0) or (len(bv_orders)>0 and bas+j-1!=bv_orders[-1]):
                    break_bv = True
                bv_orders.append(bas+j)
                new_ys.append(one[1])
        if len(bv_orders) != self.bases[-1]:
            break_bv, break_bi = True, True
        # rebuild
        self._build(nexts)
        # return
        if break_bi:
            return bv_orders, bv_orders, new_ys
        elif break_bv:
            return bv_orders, None, new_ys
        else:
            return None, None, new_ys

# synchronized with y-steps
def search_beam(models, insts, target_dict, opts, normer, sstater, para_extractor):
    xs = [i[0] for i in insts]
    xwords = [i.get_words(0) for i in insts]
    Model.new_graph()
    for _m in models:
        _m.refresh(False)
    _m0 = models[0]     # maybe common for all models, todo(warn) this inhibit certain varieties of model diff
    bsize = len(xs)
    esize_all = opts["beam_size"]
    decode_maxlen = opts["decode_len"]
    decode_maxratio = opts["decode_ratio"]
    # if need to get att-weights (todo(warn), some cohesion) (todo(WARN2) only add cov for search_beam)
    need_att = opts["decode_replace_unk"] or (opts["cov_record_mode"] != "none")
    Pruner.cov_checker = CovChecker(opts)
    decode_dump_hiddens = opts["decode_dump_hiddens"] or (opts["hid_sim_metric"] != "none")
    decode_gold_run = opts["decode_dump_hiddens"]
    # pruners
    pr_local_expand = min(opts["pr_local_expand"],esize_all)
    pr_local_diff = opts["pr_local_diff"]
    pr_local_penalty = opts["pr_local_penalty"]
    #
    pr_global_expand = min(opts["pr_global_expand"], esize_all)
    pr_global_diff = opts["pr_global_diff"]
    pr_global_penalty = opts["pr_global_penalty"]
    pr_tngram_n = opts["pr_tngram_n"]
    pr_tngram_range = opts["pr_tngram_range"]
    ##
    pr_global_nalpha = opts["pr_global_nalpha"]
    pr_global_lreward = opts["pr_global_lreward"]
    #
    remain_sizes = [esize_all for _ in range(bsize)]
    opens = [[State(sg=SearchGraph(target_dict=target_dict, src_info=insts[_nn]))] for _nn in range(bsize)]
    ends = [[] for _ in range(bsize)]
    pruned_ends = [[] for _ in range(bsize)]
    bh = BatchedHelper(opens)
    caches = []
    eos_id = target_dict.eos
    yprev = None
    for step in range(decode_maxlen):
        one_cache = []
        if step==0:
            for mi, _m in enumerate(models):
                cc = _m.start(xs)
                one_cache.append(cc)
            # todo(warn) init normer and pruner here!! (after the first step) # (#insts, ) of real lengths
            pred_lens = np.average([_m.predict_length(insts, cc=_c) for _m, _c in zip(models, one_cache)], axis=0)
            pred_lens_sigma = np.average([_m.lg.get_real_sigma() for _m in models], axis=0)
            pr_len_upper = pred_lens + opts["pr_len_khigh"] * pred_lens_sigma
            pr_len_lower = pred_lens - opts["pr_len_klow"] * pred_lens_sigma
            # for printing hiddens (same as static-training)
            if decode_gold_run:
                tooling_model = models[0]
                _, gr_ys = tooling_model.prepare_xy_(insts)
                gr_caches = one_cache.copy()
                gr_opens = [one[0] for one in opens]
                gr_ylens = [len(_y) for _y in gr_ys]
                gr_maxlen = max(gr_ylens)
                gr_yprev = None
                for gr_step in range(gr_maxlen):
                    gr_ystep, gr_mask_expr = tooling_model.prepare_y_step(gr_ys, gr_step)
                    for gr_midx, gr_m in enumerate(models):
                        if gr_step > 0:
                            gr_cc = gr_m.step(gr_caches[gr_midx], gr_yprev, None)
                            gr_caches[gr_midx] = gr_cc
                    gr_yprev = gr_ystep
                    # set up the hiddens
                    gr_hids_e = layers.BK.average([one['hid'][0]['H'] for one in gr_caches])
                    gr_hids_v = layers.BK.get_value_np(gr_hids_e)
                    gr_results_all = layers.BK.average([one["results"] for one in gr_caches])
                    gr_results_pick = layers.BK.pick_batch(gr_results_all, gr_ystep)
                    gr_results_v = layers.BK.get_value_vec(gr_results_pick)
                    for gr_j in range(bsize):
                        if gr_opens[gr_j] is not None:
                            gr_opens[gr_j].set("hidv", gr_hids_v[gr_j])
                            gr_tok, gr_score = gr_ystep[gr_j], np.log(gr_results_v[gr_j])
                            gr_next = State(prev=gr_opens[gr_j], action=Action(gr_tok, gr_score))
                            gr_next.state("GOLD")
                            if gr_tok == eos_id:
                                gr_opens[gr_j] = None
                            else:
                                gr_opens[gr_j] = gr_next
        else:
            for mi, _m in enumerate(models):
                cc = _m.step(caches[-1][mi], yprev, utils.Helper.stream_rec(opens))
                one_cache.append(cc)
        # select cands
        results = layers.BK.average([one["results"] for one in one_cache])
        this_bsize = layers.BK.bsize(results)
        results_topk = layers.BK.topk(results, pr_local_expand)
        ## need to store more values
        if need_att:
            atts_e = layers.BK.average([one["att"] for one in one_cache])
            atts_v = layers.BK.get_value_np(atts_e)
        else:
            atts_v = [None for _ in range(this_bsize)]
        #
        if decode_dump_hiddens:
            hids_e = layers.BK.average([one['hid'][0]['H'] for one in one_cache])
            hids_v = layers.BK.get_value_np(hids_e)
        else:
            hids_v = [None for _ in range(this_bsize)]
        nexts = []
        for i in range(bsize):
            # cur off if one of the criteria says it is time to end.
            # todo(warn): upper length pruner
            force_end = ((step+1) >= min(decode_maxlen, pr_len_upper[i], len(xs[i])*decode_maxratio))
            # for each one in the batch
            r_start = bh.get_basis(i)
            prev_states = opens[i]
            cur_cands = []
            cur_xsrc = xwords[i] if need_att else None
            for j, one in enumerate(prev_states):
                # for each one in the beam of one instance
                rr0 = results_topk[r_start+j]
                rr = _m0.explain_result_topkp(rr0)
                one_attv = atts_v[r_start+j]
                one_hidv = hids_v[r_start+j]
                prev_states[j].set("hidv", one_hidv)
                # todo(warn): force end for the last step
                if force_end:
                    # todo(warn): ignore eos score
                    rr = [(eos_id, 0.)]
                # local pruning
                cand_states = []
                for onep in rr:
                    cand_states.append(State(prev=prev_states[j], action=Action(onep[IDX_DIM], onep[VAL_DIM]), attw=one_attv, attention_src=cur_xsrc, _tmp_prev_idx=j))
                survive_local_cands = Pruner.local_prune(cand_states, pr_local_expand, pr_local_diff, pr_local_penalty)
                cur_cands += survive_local_cands
            # sorting them all
            cur_cands.sort(key=(lambda x: x.score_partial), reverse=True)
            # global pruning
            ok_cands = Pruner.global_prune_ngram_greedy(cand_states=cur_cands, rest_beam_size=remain_sizes[i], sig_beam_size=pr_global_expand, thresh=pr_global_diff, penalty=pr_global_penalty, ngram_n=pr_tngram_n, ngram_range=pr_tngram_range, pr_global_lreward=pr_global_lreward, pr_global_nalpha=pr_global_nalpha, )
            # prepare for next steps
            cur_nexts = []
            opens[i] = []
            for new_state in ok_cands:
                prev_inner_idx = new_state.get("_tmp_prev_idx")
                action_id = new_state.action_code
                if action_id == eos_id:
                    if new_state.length <= pr_len_lower[i]:
                        new_state.state("PR_LENGTH")
                        pruned_ends[i].append(new_state)
                    else:
                        new_state.mark_end()
                        ends[i].append(new_state)
                        remain_sizes[i] -= 1
                else:
                    new_state.state("EXPAND")
                    opens[i].append(new_state)
                    cur_nexts.append((prev_inner_idx, action_id))
            # what if we pruned away all of them -> add them back even though they are pruned
            if len(ends[i])==0 and len(cur_nexts)==0:
                for one in pruned_ends[i]:
                    one.mark_end()
                    ends[i].append(one)
                    remain_sizes[i] -= 1
            # append for one inst
            nexts.append(cur_nexts)
        # re-arrange for next step if not ending
        if sum(remain_sizes) <= 0 or sum(len(z) for z in nexts) <= 0:
            break
        bv_orders, bi_orders, new_ys = bh.rerange(nexts)
        new_caches = [_m.rerange(_c, bv_orders, bi_orders) for _c, _m in zip(one_cache, models)]
        caches.append(new_caches)
        yprev = new_ys
    # final re-ranking
    for ol in ends:
        sstater.record(ol[0].sg)
    if opts["decode_latnbest"]:
        ends = [extract_nbest(ol[0].sg, esize_all, length_reward=opts["decode_latnbest_lreward"], normalizing_alpha=opts["decode_latnbest_nalpha"], max_repeat_times=opts["decode_latnbest_rtimes"]) for ol in ends]
    normer(ends, pred_lens)
    final_list = [sorted(beam, key=lambda x: x.score_final, reverse=True) for beam in ends]
    # data.Vocab.i2w(target_dict, final_list[0][0])
    # final_list[0][0].sg.show_graph(target_dict)
    for ranked_beam in final_list:
        para_extractor.record(ranked_beam)
    return final_list

class Pruner(object):
    cov_checker = None

    # ----- for sample
    @staticmethod
    def local_prune_score(rr, maxsize, thresh, penalty):
        # nearly the same as local_prune, but on scores; return (re-normed-probs, rr_modified)
        # -- always add the max-scoring one
        rr_final = [rr[0]]
        one_score_max = rr[0][VAL_DIM]
        norm_vals = [one_score_max]
        # -- possibly pruning the rest (local pruning)
        for i, one in enumerate(rr[1:], start=1):
            if i >= maxsize:
                pass
            elif one[VAL_DIM] <= one_score_max - thresh:
                pass
            else:
                if penalty > 0.:
                    one[VAL_DIM] -= i*penalty
                rr_final.append(one)
                norm_vals.append(one[VAL_DIM])
        norm_probs = np.exp(norm_vals) / np.sum(np.exp(norm_vals), axis=0)
        return norm_probs, rr_final

    # ----- for beam
    @staticmethod
    def local_prune(cand_states, maxsize, thresh, penalty):
        # on sorted list
        # -- always add the max-scoring one
        ret = [cand_states[0]]
        one_score_max = cand_states[0].action_score()
        # -- possibly pruning the rest (local pruning)
        for i, one in enumerate(cand_states[1:], start=1):
            one_score_cur = one.action_score()
            if i >= maxsize:
                one.state("PR_LOCAL_EXPAND")
            elif one_score_cur <= one_score_max - thresh:
                one.state("PR_LOCAL_DIFF")
            else:
                if penalty > 0.:
                    one_score_cur -= i * penalty    # diversity penalty
                    one.action_score(one_score_cur) # penalty
                ret.append(one)
        return ret

    @staticmethod
    def global_prune_ngram_greedy(cand_states, rest_beam_size, sig_beam_size, thresh, penalty, ngram_n, ngram_range, pr_global_lreward=0., pr_global_nalpha=1.):
        # on sorted list, comparing according to normalized scores -- greedy pruning
        # todo: how could we compare diff length states? -> normalize partial score
        # todo: how to do pruning and sig-max (there might be crossings)? -> take the greedy way
        # _get_score_f = (lambda x: x.score_partial/x.length)
        _get_score_f = lambda x: (x.score_partial + pr_global_lreward*x.length) / (x.length ** pr_global_nalpha)
        sig_ngram_maxs = {}     # all-step max (for survived ones)
        sig_ngram_curnum = defaultdict(int)     # last-step state lists for sig
        sig_ngram_allnum = defaultdict(int)     # all-step state counts for sig
        temp_ret = []
        ngram_range = max(0, ngram_range)       # to be sure >= 0
        # pruning
        for one in cand_states:
            if len(temp_ret) >= rest_beam_size:
                one.state("PR_BEAM")
                continue
            # ngram sigs, according to the listing
            them = one.get_path(maxlen=ngram_range)
            if len(them) > 0:
                this_pruned = False
                # pruning according to sig-size and thresh
                cur_sig = one.sig_ngram(ngram_n)
                flag_not_best = False
                if cur_sig in sig_ngram_maxs:
                    this_score, high_score = _get_score_f(one), _get_score_f(sig_ngram_maxs[cur_sig])
                    if this_score <= high_score:    # not the best until current
                        flag_not_best = True
                        if sig_ngram_allnum[cur_sig] >= sig_beam_size:
                            one.state("PR_NGRAM_EXPAND")
                            this_pruned = True
                        elif this_score <= high_score - thresh:
                            one.state("PR_NGRAM_DIFF")
                            this_pruned = True
                # check cov for pruning
                if this_pruned:
                    pruner_one = sig_ngram_maxs[cur_sig]
                    if not Pruner.cov_checker.cov_ok(one, pruner_one):
                        one.state("ZZ")
                        one.tags("FAIL_PR_COV")
                        this_pruned = False
                # adding
                if not this_pruned:
                    # todo(warn) penalize here according to two criteria
                    if penalty > 0.:
                        one_score_cur = one.action_score()
                        one_score_cur -= sig_ngram_curnum[cur_sig] * penalty
                        if flag_not_best:
                            one_score_cur -= penalty
                        one.action_score(one_score_cur)
                    # add all steps for this one
                    for one_state in them:
                        one_sig = one_state.sig_ngram(ngram_n)
                        if one_sig not in sig_ngram_maxs or _get_score_f(one_state) > _get_score_f(sig_ngram_maxs[one_sig]):
                            sig_ngram_maxs[one_sig] = one_state
                        sig_ngram_allnum[one_sig] += 1
                    sig_ngram_curnum[cur_sig] += 1      # only last step
                    temp_ret.append(one)
                else:
                    pruner_one = sig_ngram_maxs[cur_sig]
                    # set pruners and record in the sg
                    one.set("PR_PRUNER", pruner_one)
                    pruner_one.add_list("PRUNING_LIST", one)
                    if one == pruner_one:
                        utils.zlog("WHAT? Self-pruning?")
            else:
                # for example, the first several steps
                temp_ret.append(one)
        return temp_ret

# dfs style searching, currently only follow the branching of the greedy one
def search_branch(models, insts, target_dict, opts, normer, sstater, para_extractor):
    # todo(warn) currently only one pass
    _get_score_f = (lambda x: x.score_partial/x.length)
    xs = [i[0] for i in insts]
    xwords = [i.get_words(0) for i in insts]
    Model.new_graph()
    for _m in models:
        _m.refresh(False)
    _m0 = models[0]     # maybe common for all models, todo(warn) this inhibit certain varieties of model diff
    bsize = len(xs)
    # params
    esize_all = opts["beam_size"]       # not traditionally beam size
    decode_maxlen = opts["decode_len"]
    decode_maxratio = opts["decode_ratio"]
    # if need to get att-weights (todo(warn))
    need_att = opts["decode_replace_unk"]
    # pruners
    pr_local_expand = min(opts["pr_local_expand"],esize_all)
    pr_local_diff = opts["pr_local_diff"]
    pr_local_penalty = opts["pr_local_penalty"]
    #
    pr_global_expand = min(opts["pr_global_expand"],esize_all)
    pr_global_diff = opts["pr_global_diff"]
    pr_global_penalty = opts["pr_global_penalty"]
    pr_tngram_n = opts["pr_tngram_n"]
    pr_tngram_range = opts["pr_tngram_range"]
    # how to select which branches to go
    branching_criter = {"abs": lambda best, current:0-current, "rel": lambda best, current:best-current,
        "b_abs": lambda best, current: (best, 0-current), "b_rel": lambda best, current: (best, best-current)}[opts["branching_criterion"]]
    branching_expand_run = opts["branching_expand_run"]
    # 0. preparations for the whole batch
    # -- opens are one for each batch, but ends are not
    opens = [State(sg=SearchGraph(target_dict=target_dict, src_info=insts[_nn])) for _nn in range(bsize)]
    ends = [[] for _ in range(bsize)]
    pruned_ends = [[] for _ in range(bsize)]
    eos_id = target_dict.eos
    # sig recombine information
    sig_ngram_maxs = [[{} for _z in range(decode_maxlen+1)] for _ in range(bsize)]
    sig_ngram_allnum = [[defaultdict(int) for _z in range(decode_maxlen+1)] for _ in range(bsize)]
    branching_points = [PriorityQueue() for _ in range(bsize)]
    # first go for a greedy run and then for several branching greedy runs
    # how to end
    branching_fullfill_ratio = opts["branching_fullfill_ratio"]
    num_run = 0
    num_newstates = [[] for _ in range(bsize)]
    while True:
        # whether to end looping
        if num_run >= esize_all:
            break_flag = True
            for tmp_idx in range(bsize):
                tmp_newstates = num_newstates[tmp_idx]
                if sum(tmp_newstates) >= branching_fullfill_ratio*tmp_newstates[0]:
                    branching_points[tmp_idx] = PriorityQueue()      # clear
                else:
                    if not branching_points[tmp_idx].empty():
                        break_flag = False
            if break_flag:
                break
        # start a new loop
        run_caches = []
        run_yprev = []
        run_unfinished = [[1 for _ in range(bsize)]]   # checking the batching information
        for false_step in range(decode_maxlen):
            one_cache = []
            if false_step==0 and num_run==0:
                for tmp_i in range(bsize):  # count: init state
                    num_newstates[tmp_i].append(1)
                for mi, _m in enumerate(models):
                    cc = _m.start(xs)
                    one_cache.append(cc)
                # todo(warn) init normer and pruner here!! (after the first step) # (#insts, ) of real lengths
                pred_lens = np.average([_m.predict_length(insts, cc=_c) for _m, _c in zip(models, one_cache)], axis=0)
                pred_lens_sigma = np.average([_m.lg.get_real_sigma() for _m in models], axis=0)
                pr_len_upper = pred_lens + opts["pr_len_khigh"] * pred_lens_sigma
                pr_len_lower = pred_lens - opts["pr_len_klow"] * pred_lens_sigma
            else:
                if false_step == 0:
                    # prepare caches and yprevs
                    combine_caches, combine_indexes = [[] for _ in models], []
                    for i in range(bsize):
                        if branching_points[i].empty():
                            run_unfinished[-1][i] = 0   # no this much branching points
                            opens[i] = None
                        else:
                            num_newstates[i].append(1)  # count: init state for branching points
                            new_state = branching_points[i].get()[-1]
                            opens[i] = new_state
                            prev_cache = new_state.get("_tmp_prev_cache")
                            utils.zcheck(len(prev_cache) == len(models), "Unequal num of caches")
                            for ii0 in range(len(models)):
                                combine_caches[ii0].append(prev_cache[ii0])
                            combine_indexes.append(new_state.get("_tmp_prev_idx"))
                            run_yprev.append(new_state.action_code)
                            new_state.state("START%s" % num_run)
                    # select and combine
                    if len(run_yprev) > 0:
                        run_caches.append([_m.recombine(_c, combine_indexes) for _m, _c in zip(models, combine_caches)])
                else:
                    for i in range(bsize):
                        num_newstates[i][-1] += run_unfinished[-1][i]   # count: continue state for branching points
                if len(run_yprev) > 0:
                    for mi, _m in enumerate(models):
                        cc = _m.step(run_caches[-1][mi], run_yprev, utils.Helper.stream_rec(opens))
                        one_cache.append(cc)
                if len(run_yprev) <= 0:     # not enough for all of them
                    break
            # get results
            results = layers.BK.average([one["results"] for one in one_cache])
            this_bsize = layers.BK.bsize(results)
            results_topk = layers.BK.topk(results, pr_local_expand)
            if need_att:
                atts_e = layers.BK.average([one["att"] for one in one_cache])
                atts_v = layers.BK.get_value_np(atts_e)
            else:
                atts_v = [None for _ in range(this_bsize)]
            # select cands and record branching points
            batched_results_base = 0
            next_unfinished = [0 for _z in range(bsize)]
            next_yids = []
            next_reorders = []
            for i in range(bsize):
                cur_state = opens[i]
                cur_len = cur_state.length
                if run_unfinished[-1][i]:
                    cur_xsrc = xwords[i] if need_att else None
                    force_end = (cur_len+1 >= min(decode_maxlen, pr_len_upper[i], len(xs[i])*decode_maxratio))
                    prev_state = opens[i]
                    rr0 = results_topk[batched_results_base]
                    rr = _m0.explain_result_topkp(rr0)
                    one_attv = atts_v[batched_results_base]
                    # todo(warn): whether expand on later runs
                    if force_end:
                        rr = [(eos_id, 0.)]
                    elif num_run < branching_expand_run:
                        pass    # allowing for expansion
                    else:
                        rr = [rr[0]]
                    # local pruning
                    cand_states = []
                    for onep in rr:
                        cand_states.append(State(prev=prev_state, action=Action(onep[IDX_DIM], onep[VAL_DIM]), attw=one_attv, attention_src=cur_xsrc, _tmp_prev_idx=batched_results_base, _tmp_prev_cache=one_cache))
                    survive_local_cands = Pruner.local_prune(cand_states, pr_local_expand, pr_local_diff, pr_local_penalty)
                    # record branching points, could only for the first run with more than 1 expansions
                    survive_best_score = survive_local_cands[0].action_score()
                    for rest_one in survive_local_cands[1:]:
                        # todo(warn): sorted(small to large for PriorityQueue) on what (id for breaking equal)
                        branching_points[i].put((branching_criter(survive_best_score, rest_one.action_score()), rest_one.id, rest_one))
                    # prepare the greedy one and record
                    new_state = cand_states[0]
                    action_id = new_state.action_code
                    # -- ngram pruning
                    this_pruned = False
                    if pr_tngram_range > 0:
                        cur_sig = new_state.sig_ngram(pr_tngram_n)
                        this_score = _get_score_f(new_state)
                        max_state = None
                        max_score = utils.Constants.MIN_V
                        number_hit = 0
                        # find max scored one for cur_sig -> todo(warn) considering the window [l-r, l+r], so also including the forwards
                        for _tmp_len in range(max(pr_tngram_n, cur_len-pr_tngram_range+1), min(cur_len+pr_tngram_range, decode_maxlen)+1):
                            if cur_sig in sig_ngram_maxs[i][_tmp_len]:
                                number_hit += sig_ngram_allnum[i][_tmp_len][cur_sig]
                                one_state = sig_ngram_maxs[i][_tmp_len][cur_sig]
                                one_score = _get_score_f(one_state)
                                if one_score > max_score:
                                    max_score = one_score
                                    max_state = one_state
                        # maybe pruning
                        if max_state is not None:
                            if this_score <= max_score:
                                if number_hit >= pr_global_expand:
                                    new_state.state("PR_NGRAM_EXPAND")
                                    this_pruned = True
                                elif this_score <= max_score - pr_global_diff:
                                    new_state.state("PR_NGRAM_DIFF")
                                    this_pruned = True
                        if number_hit > 0:
                            new_state.action_score(this_score - pr_global_penalty*number_hit)
                        # record the info
                        if not this_pruned:
                            # pass, add the sig at the last step (end)
                            pass
                        else:
                            # set pruners and record in the sg
                            new_state.set("PR_PRUNER", max_state)
                            max_state.add_list("PRUNING_LIST", new_state)
                    # ok, after pruning
                    if not this_pruned:
                        if action_id == eos_id:
                            if new_state.length <= pr_len_lower[i]:
                                new_state.state("PR_LENGTH")
                                pruned_ends[i].append(new_state)
                            else:
                                new_state.mark_end()
                                ends[i].append(new_state)
                                # add sigs
                                new_path = new_state.get_path(maxlen=pr_tngram_range)
                                for one_in_path in new_path:
                                    one_sig = one_in_path.sig_ngram(pr_tngram_n)
                                    one_len = one_in_path.length
                                    sig_ngram_allnum[i][one_len][one_sig] += 1
                                    _tmp_sig_max = sig_ngram_maxs[i][one_len]
                                    if (one_sig not in _tmp_sig_max) or _get_score_f(one_in_path) > _get_score_f(_tmp_sig_max[one_sig]):
                                        _tmp_sig_max[one_sig] = one_in_path
                        else:
                            new_state.state("EXPAND%s" % num_run)
                            opens[i] = new_state
                            next_unfinished[i] = 1
                            next_yids.append(action_id)
                            next_reorders.append(batched_results_base)
                else:
                    pass
                batched_results_base += run_unfinished[-1][i]
            # prepare next steps
            next_number = len(next_reorders)
            if next_number == 0:
                break
            run_caches.append(one_cache)
            if next_number < sum(run_unfinished[-1]):
                # shrinking
                run_caches.append([_m.rerange(_c, next_reorders, next_reorders) for _c, _m in zip(one_cache, models)])
            run_unfinished.append(next_unfinished)
            run_yprev = next_yids
        num_run += 1
    # re-pick if all pruned out by short
    # todo(warn): will there be any self-pruning?
    for i in range(bsize):
        utils.zcheck(len(ends[i]) > 0, "No end states, repick pruned ones.", func="warn")
        if len(ends[i]) == 0:
            ends[i] = pruned_ends[i]
            for one in ends[i]:
                one.mark_end()
        utils.zcheck(len(ends[i]) > 0, "No end states, currently !!??.", func="warn")
    # final re-ranking
    for ol in ends:
        sstater.record(ol[0].sg)
    if opts["decode_latnbest"]:
        ends = [extract_nbest(ol[0].sg, esize_all, length_reward=opts["decode_latnbest_lreward"], normalizing_alpha=opts["decode_latnbest_nalpha"], max_repeat_times=opts["decode_latnbest_rtimes"]) for ol in ends]
    normer(ends, pred_lens)
    final_list = [sorted(beam, key=lambda x: x.score_final, reverse=True) for beam in ends]
    for ranked_beam in final_list:
        para_extractor.record(ranked_beam)
    return final_list

def search_rerank(models, inst_pairs, target_dict, opts, normer, sstater, para_extractor):
    # inst_pairs -> (x, ys(multis), ...), golds
    adding_gold = (opts["rr_mode"] == "gold")
    Model.new_graph()
    for _m in models:
        _m.refresh(False)
    # enforce batch-size == 1
    final_list = []
    for origin_ins, gold_ins in zip(inst_pairs[0], inst_pairs[1]):
        # construct the insts
        x_idxes = origin_ins[0]
        x_words = origin_ins.get_words(0)
        insts = []
        mappings = []
        for i in range(1, len(origin_ins)):
            ys_idxes, ys_words = origin_ins[i], origin_ins.get_words(i)
            tmp_count = 0
            for one_idx, one_ws in zip(ys_idxes, ys_words):
                insts.append(data.TextInstance([x_words, one_ws], [x_idxes, one_idx]))
                mappings.append("y%s-%s"%(i-1, tmp_count))
                tmp_count += 1
        if adding_gold:
            for i in range(len(gold_ins)):
                g_idx, g_ws = gold_ins[i], gold_ins.get_words(i)
                insts.append(data.TextInstance([x_words, g_ws], [x_idxes, g_idx]))
                mappings.append("g%s"%(i-1))
        # calculations
        all_losses = []
        pred_lens = np.average([_m.predict_length(insts) for _m in models], axis=0)
        for _m in models:
            losses = _m.fb(insts, False, ret_value="losses", new_graph=False)    # by default, loss is the score: log(p)
            all_losses.append(losses)
        # construct states
        sg = SearchGraph(target_dict=target_dict, src_info=[origin_ins, gold_ins])
        start_state = State(sg=sg)
        ends = []
        for i, ins in enumerate(insts):
            file_tag = mappings[i]
            cur_state = start_state
            for j, widx in enumerate(ins[1]):
                # todo(warn): -1 * neg-log
                scores = [-1*one_losses[i][j] for one_losses in all_losses]
                avg_score = np.average(scores)
                cur_state = State(prev=cur_state, action=Action(widx, avg_score), all_scores=scores, file_tag=file_tag)
            utils.zcheck(cur_state.action_code == target_dict.eos, "Not scientific ends.")
            cur_state.mark_end()
            ends.append(cur_state)
        # normer
        normer([ends], pred_lens)
        ends_sorted = sorted(ends, key=lambda x: x.score_final, reverse=True)
        final_list.append(ends_sorted)
        # record
        ends_sorted[0].state(ends_sorted[0].get("file_tag"))
        sstater.record(sg)
    return final_list
