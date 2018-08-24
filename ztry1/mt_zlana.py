#

# analyze the lattice: lattice vs. n-best vs. gold

# lattice vs n-best
# 1. whether most of n-best are already in the lattice
# 2. whether approximation is ok
# 3. whether lattice really searches out higher scores
# 4. higher scores -> higher results: real search errors
# 5. higher scores -> lower results: model errors & fortuitous search error
# 6. lattice oracle

import numpy as np
from collections import defaultdict
import pickle, sys, copy, os, subprocess, re
import zl
from zl import utils
from zl.search2 import extract_nbest
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

printing = lambda x: print(x, flush=True)

def system(cmd, pp=True, ass=False, popen=True):
    if pp:
        printing("SYS-RUN: %s" % cmd)
    if popen:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        n = p.wait()
        output = p.stdout.read()
    else:
        n = os.system(cmd)
        output = None
    if pp:
        printing("SYS-OUT: %s" % output)
    if ass:
        assert n==0
    return output

#
class Confs(object):
    #
    MAXN = -1
    # lana1
    STEP_SCORE_THRESH = 0.001       # ignore step diff if less than this one
    # lana2
    FINAL_SCORE_THRESH = 0.001       # "EQA" if final score diff is in this range
    # lana3
    EVAL_FF = lambda fname: Confs._EVAL_FFS[Confs.EVAL_NAME](fname)
    EVAL_NAME = "ze"
    _EVAL_FFS = {"ze": lambda fname: system("perl ../../znmt/scripts/multi-bleu.perl ../../zh_en_data/Reference-for-evaluation/nist_36/nist_36.ref < %s" % (fname,)), "ed": lambda fname: system("ZMT=../.. bash ../../znmt/scripts/restore.sh < %s | perl ../../znmt/scripts/multi-bleu.perl ../../en_de_data_z5/data46.tok.de" % fname)}
    #
    # others (how to evaluate bleu)
    _smooth_f = SmoothingFunction().method2
    EVAL_RESTORE = 0

    @staticmethod
    def restore(words):
        # combine bpes of “@@”, but still not dealing with Truecase
        new_words_l = re.sub(r"(@@ )|(@@ ?$)", "", " ".join(words))
        new_words = new_words_l.split()
        return new_words

    @staticmethod
    def sbleu(refs, hyp):
        _res_ff = Confs.restore
        if Confs.EVAL_RESTORE:
            refs = [_res_ff(ws) for ws in refs]
            hyp = _res_ff(hyp)
        return sentence_bleu(refs, hyp, smoothing_function=Confs._smooth_f)

    @staticmethod
    def cbleu(l_refs, l_hyp):
        _res_ff = Confs.restore
        if Confs.EVAL_RESTORE:
            l_refs = [[_res_ff(ws) for ws in ones] for ones in l_refs]
            l_hyp = [_res_ff(ws) for ws in l_hyp]
        return corpus_bleu(l_refs, l_hyp)

# ----------
# Part 1: search analysis
# 1/2/3 n-best embedded in latice or not & approximate scoring conditions

def rec_start(length, to_sum=False):
    rec = {}
    rec["rec_len_all"] = length
    rec["rec_len_wmatch"] = 0      # word matched
    rec["rec_len_smatch"] = 0      # score matched
    rec["rec_accu_sdiff"] = 0.     # accumulated score differences
    rec["rec_num_mpoints"] = 0     # number of merged point
    rec["rec_len_wmatch_am"] = 0   # after at least one merge
    rec["rec_len_smatch_am"] = 0   # ...
    rec["rec_accu_sdiff_am"] = 0.
    if to_sum:
        rec["sum_counts"] = 0
        rec["sum_states"] = defaultdict(int)
    else:
        rec["rec_state"] = "OK"
    return rec

def rec_acckeys(rec):
    ks = []
    for k in rec:
        v = rec[k]
        if k.startswith("rec") and type(v) in [int, float]:
            ks.append(k)
    return ks

def rec_add(rec0, rec1):
    rec0["sum_counts"] += 1
    rec0["sum_states"][rec1["rec_state"]] += 1
    for k in rec_acckeys(rec1):
        if k not in rec0:
            rec0[k] = 0
        rec0[k] += rec1[k]

def rec_finish(rec):
    # first macro-average
    if "sum_counts" in rec:
        counts = rec["sum_counts"]
        for k in rec_acckeys(rec):
            rec["avg"+k] = rec[k] / counts
    # then micro-average
    rec["rec_cc_wmatch"] = rec["rec_len_wmatch"]/rec["rec_len_all"]
    rec["rec_cc_smatch"] = rec["rec_len_smatch"]/rec["rec_len_all"]
    if rec["rec_len_wmatch"] > 0:
        rec["rec_cc_sdiff"] = rec["rec_accu_sdiff"]/rec["rec_len_wmatch"]
    rec["rec_cc_wmatch_am"] = rec["rec_len_wmatch_am"]/rec["rec_len_all"]
    rec["rec_cc_smatch_am"] = rec["rec_len_smatch_am"]/rec["rec_len_all"]
    if rec["rec_len_wmatch_am"] > 0:
        rec["rec_cc_sdiff_am"] = rec["rec_accu_sdiff_am"]/rec["rec_len_wmatch_am"]
    return rec

def rec_report(rec):
    s = ""
    for n in sorted(rec.keys()):
        s += "-- " + n + ": " + str(rec[n]) + "\n"
    return s

def extract_path_from_lattice(words, sg):
    cur = sg.root
    rets = []
    prune_reason = None
    for w in words:
        # find next one
        expand = None
        for next in sg.childs(cur):
            next_w = next.descr_word()
            if next_w == w:
                expand = next
                break
        # check if it is pruned
        prune_reason = None
        if expand is None:
            prune_reason = "LOST"
        elif expand.state().startswith("PR_LOCAL") or expand.state().startswith("PR_BEAM"):
            prune_reason = expand.state()
        if prune_reason is not None:
            break
        rets.append(expand)
        # deal with merging
        if expand.state().startswith("PR_NGRAM"):
            cur = expand.get("PR_PRUNER")
        else:
            cur = expand
    return rets, prune_reason

# path: [(word, score)], sg: SearchGraph -> {}
def check_path_in_lattice(path, sg):
    SCORE_THRESH = Confs.STEP_SCORE_THRESH
    #
    rec = rec_start(len(path["words"]))
    rets, prune_reason = extract_path_from_lattice(path["words"], sg)
    #
    for i, one in enumerate(rets):
        s = path["scores"][i]
        score = one.action_score()
        sdiff = float(np.abs(score-s))
        sdiff_match = 1 if sdiff<=SCORE_THRESH else 0
        sdiff_accu = 0 if sdiff<=SCORE_THRESH else sdiff
        #
        rec["rec_len_wmatch"] += 1
        rec["rec_len_smatch"] += sdiff_match
        rec["rec_accu_sdiff"] += sdiff_accu
        if rec["rec_num_mpoints"]>0:
            rec["rec_len_wmatch_am"] += 1
            rec["rec_len_smatch_am"] += sdiff_match
            rec["rec_accu_sdiff_am"] += sdiff_accu
        # deal with merging
        if one.state().startswith("PR_NGRAM"):
            rec["rec_num_mpoints"] += 1
    if prune_reason is not None:
        rec["rec_state"] = prune_reason
    # return
    rec = rec_finish(rec)
    return rec

def analyze1(nbests, sgs):
    assert len(nbests) == len(sgs)
    rec_1 = rec_start(0, True)
    rec_n = rec_start(0, True)
    for nbs, sg in zip(nbests, sgs):
        recs = [check_path_in_lattice(one, sg) for one in nbs]
        rec_add(rec_1, recs[0])
        for r in recs:
            rec_add(rec_n, r)
    # final
    rec_finish(rec_1)
    rec_finish(rec_n)
    printing("For single-best: \n%s\n" % rec_report(rec_1))
    printing("For n-best: \n%s\n" % rec_report(rec_n))

# ----------
# Part 2: error analysis on n-best lists

# return [{...}]
# -- more general one for extracting nbest from lattice by another beam search
SEARCH_CALL_TIMES=0
def search_nbest_from_lattice(sg, best_n, sorter1, sorter2, connector):
    global SEARCH_CALL_TIMES
    SEARCH_CALL_TIMES += 1
    _TMP_KEY = "_v"+str(SEARCH_CALL_TIMES)
    stack_map = defaultdict(int)
    max_repeat_times = 1
    #
    def _set_recusively_one(one):
        _PRUNE_KEY = "PRUNING_LIST"
        assert not sg.is_pruned(one), "Not scientific to call on pruned states!!"
        if one.get(_TMP_KEY) is None:
            if one.is_start():
                v = [{"words":[], "scores":[], "length":0,"score_sum":0.,"score_final":0.}]
                one.set(_TMP_KEY, v)
            else:
                id_sets = {id(one)}    # todo(warn): repeated states?
                combined_list = [one]
                tmp_list = one.get(_PRUNE_KEY)
                if tmp_list is not None:
                    for zzz in tmp_list:
                        zzz_id = id(zzz)
                        if zzz_id not in id_sets:
                            id_sets.add(zzz_id)
                            combined_list.append(zzz)
                        else:   # surprised
                            pass
                            # printing("Surprised of repeated states!")
                # combined_list = one.get(_PRUNE_KEY)
                # if combined_list is None:
                #     combined_list = []
                for _one in combined_list:
                    assert _one.get(_TMP_KEY) is None, "Not scientific pruning states!!"
                #
                cands = []
                for pp in combined_list:
                    link_sig = pp.sig_idlink()
                    if stack_map[link_sig] >= max_repeat_times:
                        pass
                    else:
                        stack_map[link_sig] += 1
                        _set_recusively_one(pp.prev)
                        stack_map[link_sig] -= 1
                        for one_prev in pp.prev.get(_TMP_KEY):
                            new_one = connector(one_prev, pp)
                            cands.append(new_one)
                v = sorted(cands, key=sorter1, reverse=True)[:best_n]
                one.set(_TMP_KEY, v)    # not on pruned states
    #
    final_cands = []
    for one in sg.get_ends():
        _set_recusively_one(one)
        vs = one.get(_TMP_KEY)
        final_cands += vs
    v = sorted(final_cands, key=sorter2, reverse=True)[:best_n]
    return v

#
def ff_scorer_nonorm(c):
    return c["score_sum"]+c["length"]*1.0

def ff_scorer_norm(c):
    return (c["score_sum"]+c["length"]*1.0)/c["length"]

def set_sbleu(c, refs):
    if "sbleu" not in c:
        sb = Confs.sbleu(refs, c["words"])
        c["sbleu"] = sb

# n-best according to model score
# todo(warn): fix extracting method
def obtain_nbest_from_lattice(sg, best_n, norm):
    def connector(cprev, pp):
        ret = copy.deepcopy(cprev)
        ret["words"].append(pp.descr_word())
        ret["scores"].append(pp.action_score())
        ret["length"] += 1
        ret["score_sum"] += ret["scores"][-1]
        return ret
    list0 = search_nbest_from_lattice(sg, best_n, ff_scorer_nonorm, ff_scorer_nonorm, connector)
    if norm:    # final norm
        list0.sort(key=ff_scorer_norm, reverse=True)
    return list0

# n-best according to partial sBLEU, and final oracle as sBLEU
# refs: list of list of strs
def obtain_oracle_from_lattice(sg, best_n, refs):
    # calculate stats for refs
    refs_maps = defaultdict(int)
    for ref in refs:
        # count one
        len_r = len(ref)
        for ngram in range(4):
            mm = defaultdict(int)
            for start in range(0, len_r-ngram):
                sig = "//".join(ref[start:start+ngram+1])
                mm[sig] += 1
            # merge one
            for n in mm:
                refs_maps[n] = max(refs_maps[n], mm[n])
    #
    def sorter1(c):
        return c["pb"]  # partial BLEU
    def sorter2(c):
        set_sbleu(c, refs)
        return c["sbleu"]
    def connector(cprev, pp):
        ret = copy.deepcopy(cprev)
        ret["words"].append(pp.descr_word())
        ret["scores"].append(pp.action_score())
        ret["length"] += 1
        ret["score_sum"] += ret["scores"][-1]
        #
        if "pb" not in ret:
            # start one
            ret["clipped_maps"] = defaultdict(int)
            ret["matches"] = [0] * 4
        pb = 0
        for ngram in range(4):
            a = b = 0.1     # 0.1 as smoothing
            if len(ret["words"]) >= ngram+1:
                sig = "//".join(ret["words"][-ngram-1:])
                if refs_maps[sig]>0:
                    ret["clipped_maps"][sig] += 1
                    if ret["clipped_maps"][sig] <= refs_maps[sig]:
                        ret["matches"][ngram] += 1
                a += ret["matches"][ngram]
                b += len(ret["words"]) - ngram
            pb += np.log(a/b)
        pb /= 4
        ret["pb"] = pb
        return ret
    return search_nbest_from_lattice(sg, best_n, sorter1, sorter2, connector)

# return flags
# sbleu: SB_SAME, SB_SGBETTER, SB_NBBETTER || score: SC_EQA, SC_SGHIGH, SC_NBHIGH ||
#   (when NBHIGH):
def ana_nbests(sg, nb_sg, nb1, refs, ff_scorer):
    for one in nb_sg:
        one["score_final"] = ff_scorer(one)
        set_sbleu(one, refs)
    for one in nb1:
        one["score_final"] = ff_scorer(one)
        set_sbleu(one, refs)
    nb_sg.sort(key=lambda c: c["score_final"], reverse=True)
    nb1.sort(key=lambda c: c["score_final"], reverse=True)
    #
    best_sg = nb_sg[0]
    best_nb1 = nb1[0]
    if best_sg["words"] == best_nb1["words"]:
        flag_sbleu = "SB_SAME"
        flag_score = "SC_EQA"
    else:
        if best_sg["sbleu"] > best_nb1["sbleu"]:
            flag_sbleu = "SB_SGBETTER"
        elif best_sg["sbleu"] < best_nb1["sbleu"]:
            flag_sbleu = "SB_NBBETTER"
        else:
            flag_sbleu = "SB_SAME2"
        if np.abs(best_sg["score_final"]-best_nb1["score_final"]) <= Confs.FINAL_SCORE_THRESH:
            flag_score = "SC_EQA2"
        elif best_sg["score_final"] > best_nb1["score_final"]:
            flag_score = "SC_SGHIGH"
        else:
            flag_score = "SC_NBHIGH"
    flags = [flag_score, flag_sbleu, flag_score+"|"+flag_sbleu]
    # especially when nb1 has higher score?
    if flag_score == "SC_NBHIGH":
        flag_sp1 = None
        for rest in nb_sg[1:]:
            if rest["words"] == best_nb1["words"]:
                flag_sp1 = "INNB"
        if flag_sp1 is None:
            _, reason = extract_path_from_lattice(best_nb1["words"], sg)
            flag_sp1 = "INSG_%s" % reason
        flags.append(flag_sp1)
        flags.append(flag_sbleu+"|"+flag_sp1)
    return flags

# -> analyzing the sg, nbest from sg, the-other-nbest
def analyze2(nbests, sgs, refs_words, norm):
    assert len(nbests) == len(sgs)
    assert len(nbests) == len(refs_words)
    rec = defaultdict(int)
    sg_nbs = []
    for nb1, sg, refs in zip(nbests, sgs, refs_words):
        nb_sg_one = obtain_nbest_from_lattice(sg, 10, norm)
        _ff_scorer = ff_scorer_norm if norm else ff_scorer_nonorm
        flags = ana_nbests(sg, nb_sg_one, nb1, refs, _ff_scorer)
        for f in flags:
            rec[f] += 1
        sg_nbs.append(nb_sg_one)
    printing("For nbest analysis with NORM=%s: \n%s\n" % (norm, rec_report(rec)))
    return sg_nbs

# ----------
# Part 3: oracle & overall analysis

def calc_bleus(ones, refs, name):
    #
    printing("Eval for %s." % name)
    with open(name+".txt", "w") as fd:
        for one in ones:
            fd.write("%s\n" % " ".join(one["words"][:-1]))
    Confs.EVAL_FF(name+".txt")
    n_sent, n_ref = len(refs), len(refs[0])
    cb = Confs.cbleu(refs, [z["words"] for z in ones])
    printing("Bleu with unk: %s" % cb)

def argmax(ll, key):
    best_idx = 0
    best_val = key(ll[0])
    for idx in range(1, len(ll)):
        v = key(ll[idx])
        if v>best_val:
            best_idx = idx
            best_val = v
    return best_idx

def count_states(sg):
    _TMP_KEY = "counts"
    _PRUNE_KEY = "PRUNING_LIST"
    stack_map = defaultdict(int)
    states_map = defaultdict(list)
    def _add_map(s):
        l = s.length
        if s not in states_map[l]:
            states_map[l].append(s)
    def _count(one):
        # _add_map(one)
        if one.get(_TMP_KEY) is None:
            if one.is_start():
                one.set(_TMP_KEY, 1)
            else:
                id_sets = {id(one)}
                combined_list = [one]
                tmp_list = one.get(_PRUNE_KEY)
                if tmp_list is not None:
                    for zzz in tmp_list:
                        zzz_id = id(zzz)
                        if zzz_id not in id_sets:
                            id_sets.add(zzz_id)
                            combined_list.append(zzz)
                        else:   # surprised
                            pass
                c = 0
                for z in combined_list:
                    # _add_map(z)
                    link_sig = z.sig_idlink()
                    if stack_map[link_sig] <= 0:
                        stack_map[link_sig] += 1
                        c += _count(z.prev)
                        stack_map[link_sig] -= 1
                counts = c
                one.set(_TMP_KEY, counts)
        return one.get(_TMP_KEY)
    #
    all_c = [_count(z) for z in sg.get_ends()]
    all_counts = sum(all_c)
    return all_counts

def analyze3(nbests, sgs, refs_words, sg_nbs_norm, sg_nbs_nonorm):
    _get_first = lambda ones: [z[0] for z in ones]
    _get_oracle = lambda ones: [z[argmax(z, lambda c:c["sbleu"])] for z in ones]
    # extract oracle from sg
    sg_oracles_nbs = [obtain_oracle_from_lattice(sg, 10, refs) for sg,refs in zip(sgs,refs_words)]
    sg_counts = [count_states(sg) for sg in sgs]
    printing("FIRSTLY, average state counts is %s" % (np.average(sg_counts),))
    sg_oracles = _get_first(sg_oracles_nbs)
    calc_bleus(sg_oracles, refs_words, "sg_oracle")
    printing("\n")
    #
    for norm, sg_nbs in zip([False, True], [sg_nbs_nonorm, sg_nbs_norm]):
        n_suffix = "norm" if norm else "nonorm"
        ff_scorer = ff_scorer_norm if norm else ff_scorer_nonorm
        printing("Ana3 for %s" % n_suffix)
        #
        for ones in nbests:
            for one in ones:
                one["score_final"] = ff_scorer(one)
            ones.sort(key=lambda c: c["score_final"], reverse=True)
        # get oracles from lists
        nbs_oracles = _get_oracle(nbests)
        calc_bleus(nbs_oracles, refs_words, "nbs_oracles")
        sg_nbs_oracles = _get_oracle(sg_nbs)
        calc_bleus(sg_nbs_oracles, refs_words, "sg_nbs_oracles")
        nbs_fisrt = _get_first(nbests)
        calc_bleus(nbs_fisrt, refs_words, "nbs_fisrt")
        sg_nbs_first = _get_first(sg_nbs)
        calc_bleus(sg_nbs_first, refs_words, "sg_nbs_first")
        #
        rec = defaultdict(int)
        for cur_sg_oracle, cur_nbs_oracle, cur_sg_nbs_oracle, cur_nbs_fisrt, cur_sg_nbs_first in zip(sg_oracles, nbs_oracles, sg_nbs_oracles, nbs_fisrt, sg_nbs_first):
            # which oracle is better
            zlist = [cur_sg_oracle, cur_nbs_oracle, cur_sg_nbs_oracle, cur_nbs_fisrt, cur_sg_nbs_first]
            zlist_names = ["SG", "NBS", "SG_NBS", "NBS1", "SG_NBS1"]
            cur_bidx = argmax(zlist, lambda c:c["sbleu"])
            best_name = "OB_"+zlist_names[cur_bidx]
            best_one = zlist[cur_bidx]
            rec[best_name] += 1
            # oracle hits
            for one, one_name in zip(zlist, zlist_names):
                if one["words"] == best_one["words"]:
                    hit_name = "HIT_"+one_name
                    hit_name2 = best_name + "_" + hit_name
                    rec[hit_name] += 1
                    rec[hit_name2] += 1
        printing("For oracle analysis with NORM=%s: \n%s\n" % (norm, rec_report(rec)))

# ----------
# Finally: all together

# specific nbests format
# {"words":,"scores":,"length":,"score_sum":,"score_final":,}
def load_nbest(file, maxn=-1):
    ret = []
    with open(file) as fd:
        ones = []
        for line in fd:
            fields = line.split()
            if len(fields) == 0:
                if len(ones) > 0:
                    ret.append(ones)
                    ones = []
                    if len(ret) == maxn:
                        break
            else:
                ending = fields[-1].split("FINALLY")
                assert len(ending)==2
                fields = fields[:-1] + [ending[0]]
                words, scores = [], []
                for ss in fields:
                    thems = ss.split("|")
                    cur_word = "|".join(thems[:-1])
                    cur_score = float(thems[-1])
                    words.append(cur_word)
                    scores.append(cur_score)
                ending_fields = ending[1].split("|")
                length, score_sum, score_final = int(ending_fields[-3]), float(ending_fields[-2]), float(ending_fields[-1])
                d = {"words":words,"scores":scores,"length":length,"score_sum":score_sum,"score_final":score_final,}
                ones.append(d)
        if len(ones) > 0:
            ret.append(ones)
    printing("load nbest from %s, #%d." % (file, len(ret)))
    return ret

def load_sg(file, maxn=-1):
    sgs = []
    with open(file, "rb") as fd:
        try:
            while True:
                one = pickle.load(fd)
                sgs.append(one)
                if len(sgs) == maxn:
                    break
        except EOFError:
            pass
    printing("load sgs from %s, #%d." % (file, len(sgs)))
    return sgs

def main():
    for tok in sys.argv[4:]:
        fields = tok.split(":")
        if len(fields)==2:
            n,v = fields
            if hasattr(Confs, n):
                printing("Setting %s to %s." % (n,v))
                realv = type(getattr(Confs, n))(v)
                setattr(Confs, n, realv)
    MAXN=Confs.MAXN
    nbs = load_nbest(sys.argv[1], MAXN)
    sgs = load_sg(sys.argv[2], MAXN)
    refs = load_nbest(sys.argv[3], MAXN)
    refs_words = [[r2["words"] for r2 in r] for r in refs]
    # for debugging
    if True:
        de_bests = []
        for sg in sgs:
            # todo(warn): previous bug
            them = sg.bfs()
            for s0, s1 in them:
                for one in s0+s1:
                    zlist = one.get("PRUNING_LIST")
                    if zlist is not None:
                        if one in zlist:
                            zlist.remove(one)
            de_nb = extract_nbest(sg, 10, length_reward=1.0)
            de_nb.sort(key=lambda x: x.score_partial/x.length, reverse=True)
            de_bests.append({"words":de_nb[0].show_words(td=sg.target_dict)})
        calc_bleus(de_bests, refs_words, "debug_sp")
    #
    printing("Start Analyze 1.")
    analyze1(nbs, sgs)
    printing("Start Analyze 2 without norm.")
    sg_nbs_nonorm = analyze2(nbs, sgs, refs_words, False)
    printing("Start Analyze 2 with norm.")
    sg_nbs_norm = analyze2(nbs, sgs, refs_words, True)
    printing("Start Analyze 3.")
    analyze3(nbs, sgs, refs_words, sg_nbs_norm, sg_nbs_nonorm)

if __name__ == '__main__':
    utils.Checker.init(True)
    main()

# PYTHONPATH=$DY_ZROOT/cbuild/python:../../znmt python3 -m pdb ../../znmt/ztry1/mt_zlana.py

# en_zh
# for n in 10 30 50 100; do
# PYTHONPATH=$DY_ZROOT/cbuild/python:../../znmt python3 ../../znmt/ztry1/mt_zlana.py z.nist_36.0b-${n}.nbests z.nist_36.ana-10.sg ../decode_gold/gold.out.nbests MAXN:-1 |& tee log_zlana_sg10vsnb${n}_0217
# done

# en_de (z.10.bpe, gold.out)
# PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ../../znmt/test.py -v --test_batch_size 8 -o z.10.bpe -t ../../en_de_data_z5/data46.bpe.en ../../en_de_data_z5/data46.tok.de -d ../../baselines/ed_ev/{"src","trg"}.v -m ../../baselines/ed_ev/zbest.model --dynet-devices GPU:5 --pr_local_diff 2.3 --normalize_way add --normalize_alpha 0.4 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.4 --decode_latnbest --decode_latnbest_lreward 0.4 --decode_dump_sg -k 10
# PYTHONPATH=$DY_ZROOT/gbuild/python python3.5 ../../znmt/rerank.py -v --eval_metric bleu -o gold.out -t ../../en_de_data_z5/data46.bpe.en ../../en_de_data_z5/data46.bpe.de.list --gold ../../en_de_data_z5/data46.tok.de -d ../../baselines/ed_ev/{"src","trg"}.v --dynet-devices GPU:5 -m ../../baselines/ed_ev/zbest.model
#
# for n in 10 30 50 100; do
# PYTHONPATH=$DY_ZROOT/cbuild/python:../../znmt python3 ../../znmt/ztry1/mt_zlana.py z.data46.n4-${n}.bpe.nbests z.10.bpe.sg gold.out.nbests MAXN:-1 EVAL_NAME:ed |& tee log_zlana_0217_sg10vsnb${n}_r0
# PYTHONPATH=$DY_ZROOT/cbuild/python:../../znmt python3 ../../znmt/ztry1/mt_zlana.py z.data46.n4-${n}.bpe.nbests z.10.bpe.sg gold.out.nbests MAXN:-1 EVAL_NAME:ed EVAL_RESTORE:1 |& tee log_zlana_0217_sg10vsnb${n}_r1
# done
