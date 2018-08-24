# could be seen as reranked and analyzed by oracles

from zl import utils
from collections import defaultdict
import math
import numpy as np

# --- analysis
# to calculate bleus, and also collecting relative information
# partially from NLTK-BLEU: https://github.com/nltk/nltk/blob/develop/nltk/translate/bleu_score.py
class BleuCalculator(object):
    @staticmethod
    def count_ngrams(ll, n):
        # list of {}
        ret = []
        for i in range(n):
            mm = defaultdict(int)
            for start in range(0, len(ll)-i):
                sig = "|".join([str(s) for s in ll[start:start+i+1]])
                mm[sig] += 1
            ret.append(mm)
        return ret

    @staticmethod
    def closest_ref_length(ref_lens, hyp_len):
        closest_ref_len = min(ref_lens, key=lambda ref_len:(abs(ref_len - hyp_len), ref_len))
        return closest_ref_len

    # step1: count (add hit countf to each instance: "stat"->[[near-length, 1, 2, 3, 4]])
    @staticmethod
    def add_clipped_counts(golds, preds, n, on_words=False):
        if on_words:
            get_list_ff = lambda x, i: x.get_words(i)
        else:
            # remember to remove EOS
            get_list_ff = lambda x, i: x[i][:-1]
        # preds is list of TextInstance(multi), golds is list of TextInstance as the References
        for ones in preds:
            utils.zcheck_matched_length(ones, golds)
        # countings
        for ones in preds:
            for one_pred, one_gold in zip(ones, golds):
                l_pred, l_gold = len(one_pred), len(one_gold)
                # count gold ngrams
                ngrams_gold = [defaultdict(int) for _ in range(n)]
                lengths_gold = []
                for i in range(l_gold):
                    one_list = get_list_ff(one_gold, i)
                    lengths_gold.append(len(one_list))
                    ngrams_one_gold = BleuCalculator.count_ngrams(one_list, n)
                    for onegram_gold, onegram_one_gold in zip(ngrams_gold, ngrams_one_gold):
                        for k in onegram_one_gold:
                            onegram_gold[k] = max(onegram_gold[k], onegram_one_gold[k])
                # count pred ngrams and store
                infos, infos2, sbs = [], [], []
                for i in range(l_pred):
                    one_list = get_list_ff(one_pred, i)
                    ngrams_one_pred = BleuCalculator.count_ngrams(one_list, n)
                    # the informations to be collected (0:nearest length, 1,2,3,4, ngram-match-count)
                    crl = BleuCalculator.closest_ref_length(lengths_gold, len(one_list))
                    info, info2 = [crl], [len(one_list)]
                    for _i in range(n):
                        info2.append(max(info2[0]-_i, 0))
                    for onegram_gold, onegram_one_pred in zip(ngrams_gold, ngrams_one_pred):
                        cc = 0
                        for k in onegram_one_pred:
                            cc += min(onegram_gold[k], onegram_one_pred[k])
                        info.append(cc)
                    one_sb = BleuCalculator.bleu4(info[0], info2[0], info[1:], info2[1:], smoothing=True, ss_short=True)
                    sbs.append(one_sb)
                    infos.append(info)
                    infos2.append(info2)
                one_pred.set("stat", infos)
                one_pred.set("stat2", infos2)
                one_pred.set("sb", sbs)

    # real calculations and analysis and rankings: based on the stats of instances
    @staticmethod
    def brevity_penalty(closest_ref_len, hyp_len):
        if hyp_len > closest_ref_len:
            return 1
        # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
        elif hyp_len == 0:
            return 0
        else:
            return math.exp(1 - closest_ref_len / hyp_len)

    @staticmethod
    def bleu4(length_gold, length_pred, counts_hit, counts_all, smoothing=False, report=False, ss_short=False):
        s, cc = 0., 0
        them = {}
        bp = BleuCalculator.brevity_penalty(length_gold, length_pred)
        them["bp"] = bp
        utils.zcheck_matched_length(counts_hit, counts_all)
        for h, a in zip(counts_hit, counts_all):
            if cc>0 and smoothing:
                # +1 smooth for n>1 maybe # todo(warn) may result in 0/*/*/*
                vv = (h+1)/(a+1)
            else:
                vv = h/a
            them[cc] = vv
            if vv <= 0:
                utils.zlog("Zero 1-gram counts !!", func="warn")
                s += utils.Constants.MIN_V
            else:
                s += math.log(vv)
            cc += 1
        s /= cc
        bleu = bp * math.exp(s)
        them["bleu"] = bleu
        ss = None
        if report:
            # utils.zlog("BLEU4-Counts: %s-%s" % (counts_hit, counts_all))
            ss = "BLEU = %.2f, %.1f/%.1f/%.1f/%.1f (BP=%.3f, hyp_len=%d, ref_len=%d)" \
                  % (them["bleu"]*100, them[0]*100, them[1]*100, them[2]*100, them[3]*100, bp, length_pred, length_gold)
            utils.zlog(ss)
        if ss_short:
            ss = "%.2f(BP=%.3f,L=%d)" % (them["bleu"]*100, bp, length_pred)
        return bleu, ss

    @staticmethod
    def _argmax_list(ll, amount=10, cfs=(lambda x: x[0]>1e-5, lambda x: x[0]<=1e-5 and x[0]>=-1e-5, lambda x: x[0]<-1e-5), buckets=(0, 15, 30, 50, 1000), golds_len0=None):
        def _sort_and_report(_l2):
            rankings = sorted(_l2, key=lambda x: x[-1], reverse=True)
            for i in range(min(r, amount)):
                idx, content = rankings[i][0], rankings[i][1]
                utils.zlog("#%d Max-ranking, index is %s, content is %s." % (i, idx, content), func="details")
        # --
        r = len(ll)
        rs = []
        rs_descr = []
        for cf in cfs:
            cc = sum(1 if cf(one) else 0 for one in ll)
            rs.append(cc)
            rs_descr.append("%d/%d/%.3f" % (cc, r, cc/r))
        utils.zlog("Countings: %s" % (" ".join(rs_descr),))
        _sort_and_report([(i,one) for i,one in enumerate(ll)])
        # analyzing buckets of length of gold0
        for i in range(len(buckets)-1):
            a, b = buckets[i], buckets[i+1]
            _rf = lambda x: a <= x and x < b
            ll2 = []
            for i, one in enumerate(ll):
                if _rf(golds_len0[i]):
                    ll2.append((i, one))
            utils.zlog("Range [%d, %d): %d/%d/%.3f" % (a, b, len(ll2), r, len(ll2)/r), func="details")
            _sort_and_report(ll2)

    @staticmethod
    def _cmp(ll):
        # return max idx, -1 if all equal
        max_idx = 0
        max_value = ll[0]
        equal_flag = True
        if len(ll) > 1:
            for i, one in enumerate(ll[1:]):
                if one > max_value:
                    max_idx = i+1
                    max_value = one
                    equal_flag = False
                elif one < max_value:
                    equal_flag = False
        if equal_flag:
            return None
        else:
            return max_idx

    # step 2: add sentence-level bleu and rankings to instances
    @staticmethod
    def analyse_single(golds, pred, kbest, n):
        # sentence-level bleu score & ranking (average or max)
        len_inst = len(pred)
        golds_len0 = [len(g[0]) for g in golds]
        # sentence level smoothed bleu score
        # only consider the first k items in the list
        # t1: averaged ones
        utils.zlog("t1: Averages")
        # -- t10: average sent-bleu
        cum = 0.
        for p in pred:
            lp = len(p)
            cc = min(kbest, lp)
            cum += sum([z[0] for z in p.get("sb")[:cc]]) / cc
        utils.zlog("t10: Average sentence BLEU of kbest(k=%s) ones: BLEU=%.3f" % (kbest, cum/len_inst))
        # -- t11: average corpus-bleu
        count = 0
        cum1, cum2 = [0]*(n+1), [0]*(n+1)
        for p in pred:
            lp = len(p)
            cc = min(kbest, lp)
            count += cc
            s1, s2 = p.get("stat"), p.get("stat2")
            for i in range(cc):
                utils.Helper.add_inplace_list(cum1, s1[i])
                utils.Helper.add_inplace_list(cum2, s2[i])
        utils.zlog("t11: Average corpus BLEU of kbest(k=%s) ones, but average count is %.3f." % (kbest, count/len_inst))
        BleuCalculator.bleu4(cum1[0], cum2[0], cum1[1:], cum2[1:], report=True)
        # t2: oracle max (best sentence BLEU): influence on corpus-one-bleu
        utils.zlog("t2: About oracle max")
        # -- t20: how many is already oracle max as max & how much improvement of oracle-max
        # --- pred best
        p_cum1, p_cum2 = [0]*(n+1), [0]*(n+1)
        o_cum1, o_cum2 = [0]*(n+1), [0]*(n+1)
        hit_counts = 0
        obest_ranks = []
        for p in pred:
            lp = len(p)
            cc = min(kbest, lp)
            s1, s2, sbs = p.get("stat"), p.get("stat2"), p.get("sb")
            obest = int(np.argmax([z[0] for z in sbs[:cc]]))
            # record oracle best
            obest_ranks.append(obest)
            if obest == 0:
                hit_counts += 1
            # record pred one
            utils.Helper.add_inplace_list(p_cum1, s1[0])
            utils.Helper.add_inplace_list(p_cum2, s2[0])
            # record oracle one
            utils.Helper.add_inplace_list(o_cum1, s1[obest])
            utils.Helper.add_inplace_list(o_cum2, s2[obest])
        utils.zlog("t20: oracle hit is %s/%s/%.3f; prediction & oracle" % (hit_counts, len_inst, hit_counts/len_inst))
        bleu_base, _ = BleuCalculator.bleu4(p_cum1[0], p_cum2[0], p_cum1[1:], p_cum2[1:], report=True)
        BleuCalculator.bleu4(o_cum1[0], o_cum2[0], o_cum1[1:], o_cum2[1:], report=True)
        # --
        if kbest > 1:
            sbleu_improves = []     # list of (score, p[0], p[oracle]) -> sentence bleu improves
            cbleu_improves = []     # list of (score, final) -> corpus bleu improves after replacing
            for oidx, p in zip(obest_ranks, pred):
                s1, s2, sbs = p.get("stat"), p.get("stat2"), p.get("sb")
                sbleu_improves.append((sbs[oidx][0]-sbs[0][0], sbs[0][-1], sbs[oidx][-1], "Oracle-Rank %s"%oidx))
                repl_cum1, repl_cum2 = p_cum1.copy(), p_cum2.copy()
                utils.Helper.add_inplace_list(repl_cum1, s1[0], -1)
                utils.Helper.add_inplace_list(repl_cum1, s1[oidx], 1)
                utils.Helper.add_inplace_list(repl_cum2, s2[0], -1)
                utils.Helper.add_inplace_list(repl_cum2, s2[oidx], 1)
                bleu_change = BleuCalculator.bleu4(repl_cum1[0], repl_cum2[0], repl_cum1[1:], repl_cum2[1:])
                cbleu_improves.append((bleu_change[0]-bleu_base, bleu_change[-1], sbs[0][-1], sbs[oidx][-1]))
            # -- t21: which ones improves most (sbleu) if replaced by oracle-max
            utils.zlog("t21: improves at sentence bleus")
            BleuCalculator._argmax_list(sbleu_improves, golds_len0=golds_len0)
            # -- t22: which ones improves most (replace-cbleu) if replaced by oracle-max
            utils.zlog("t22: improves at corpus bleus with replacing")
            BleuCalculator._argmax_list(cbleu_improves, golds_len0=golds_len0)
        # t3: improvements by gold (once enough: thus only when kbest==1)
        if kbest == 1:
            utils.zlog("t3: About gold replacing")
            sbleu_goldimpr = []     # list of (score, p[0]) -> sentence bleu improves
            cbleu_goldimpr = []     # list of (score, final) -> corpus bleu improves after replacing
            for p in pred:
                s1, s2, sbs = p.get("stat"), p.get("stat2"), p.get("sb")
                sbleu_goldimpr.append((1.0-sbs[0][0], sbs[0][-1]))
                repl_cum1, repl_cum2 = p_cum1.copy(), p_cum2.copy()
                replace_counts = [s2[0][0], ] + [max(0, s2[0][0]-i) for i in range(n)]
                utils.Helper.add_inplace_list(repl_cum1, s1[0], -1)
                utils.Helper.add_inplace_list(repl_cum1, replace_counts, 1)
                utils.Helper.add_inplace_list(repl_cum2, s2[0], -1)
                utils.Helper.add_inplace_list(repl_cum2, replace_counts, 1)
                bleu_change = BleuCalculator.bleu4(repl_cum1[0], repl_cum2[0], repl_cum1[1:], repl_cum2[1:])
                cbleu_goldimpr.append((bleu_change[0]-bleu_base, bleu_change[-1], sbs[0][-1]))
            # -- t23: which ones improves most (sbleu) if replaced by gold
            utils.zlog("t31: gold_comparing at sentence bleus")
            BleuCalculator._argmax_list(sbleu_goldimpr, golds_len0=golds_len0)
            # -- t24: which ones improves most (replace-cbleu) if replaced by gold
            utils.zlog("t32: gold_comparing at corpus bleus with replacing")
            BleuCalculator._argmax_list(cbleu_goldimpr, golds_len0=golds_len0)

    @staticmethod
    def analyse_multi(golds, preds, kbest, n):
        # todo(warn): only compares the pred[0]
        golds_len0 = [len(g[0]) for g in golds]
        if kbest > 1:
            pass
        else:
            # which one get the best max results
            utils.zlog("mt0: About comparing the predicted best ones")
            # - first get best sbleu (notice that this only takes the p[0], and the index is on the preds list)
            their_cums1 = [[0]*(n+1) for _ in range(len(preds))]
            their_cums2 = [[0]*(n+1) for _ in range(len(preds))]
            max_idxes = []
            num_pred = len(preds)
            len_inst = len(preds[0])
            for i in range(len_inst):
                their_results = []
                for j in range(num_pred):
                    p = preds[j][i]
                    s1, s2, sbs = p.get("stat"), p.get("stat2"), p.get("sb")
                    utils.Helper.add_inplace_list(their_cums1[j], s1[0])
                    utils.Helper.add_inplace_list(their_cums2[j], s2[0])
                    their_results.append(sbs[0])
                # argmax
                max_idx = BleuCalculator._cmp(their_results)
                max_idxes.append(max_idx)
            num_equal = sum(1 if one is None else 0 for one in max_idxes)
            utils.zlog("Specifically, equal rate is: %d/%d/%.3f" % (num_equal, len_inst, num_equal/len_inst))
            # analyzing for each preds
            for j in range(num_pred):
                num_hit = sum(1 if j==one else 0 for one in max_idxes)
                num_good = num_hit + num_equal
                utils.zlog("Specifically, for file #%s: %d/%d(%.3f)/%d(%.3f)" % (j, num_hit, len_inst, num_hit/len_inst, num_good, num_good/len_inst))
                bleu_base, _ = BleuCalculator.bleu4(their_cums1[j][0], their_cums2[j][0], their_cums1[j][1:], their_cums2[j][1:], report=True)
                sbleu_improves = []     # list of (score, p[0], best[0]) -> sentence bleu improves
                cbleu_improves = []     # list of (score, final) -> corpus bleu improves after replacing
                cur_ii = 0
                for oidx, p in zip(max_idxes, preds[j]):
                    if oidx is None:    # equal, count as self
                        oidx = j
                    pbest = preds[oidx][cur_ii]
                    s1, s2, sbs = p.get("stat"), p.get("stat2"), p.get("sb")
                    b1, b2, bbs = pbest.get("stat"), pbest.get("stat2"), pbest.get("sb")
                    if_in_idx = None
                    for _i, _item in enumerate(sbs):
                        if bbs[0] == _item:
                            if_in_idx = _i
                    sbleu_improves.append((bbs[0][0]-sbs[0][0], sbs[0][-1], bbs[0][-1], "Here-Rank %s" % if_in_idx))
                    repl_cum1, repl_cum2 = their_cums1[j].copy(), their_cums2[j].copy()
                    utils.Helper.add_inplace_list(repl_cum1, s1[0], -1)
                    utils.Helper.add_inplace_list(repl_cum1, b1[0], 1)
                    utils.Helper.add_inplace_list(repl_cum2, s2[0], -1)
                    utils.Helper.add_inplace_list(repl_cum2, b2[0], 1)
                    bleu_change = BleuCalculator.bleu4(repl_cum1[0], repl_cum2[0], repl_cum1[1:], repl_cum2[1:])
                    cbleu_improves.append((bleu_change[0]-bleu_base, bleu_change[-1], sbs[0][-1]))
                    cur_ii += 1
                # -- mt01: which ones improves most (sbleu) if replaced by best-sbleu one
                utils.zlog("mt01: best_comparing at sentence bleus (improves from this one to the best)")
                BleuCalculator._argmax_list(sbleu_improves, golds_len0=golds_len0)
                # -- mt02: which ones improves most (replace-cbleu) if replaced by best-sbleu one
                utils.zlog("mt02: best_comparing at corpus bleus with replacing (improves from this one to the best)")
                BleuCalculator._argmax_list(cbleu_improves, golds_len0=golds_len0)

    # out
    @staticmethod
    def analyse(srcs, golds, preds, kbests, n=4, on_words=True):
        # mainly two goals: compare pred/oracle/gold & compare between preds
        BleuCalculator.add_clipped_counts(golds, preds, n, on_words=on_words)
        for curk in kbests:
            utils.zlog("Start for kbest: k==%s" % curk, func="time")
            for i, pred in enumerate(preds):
                utils.zlog("For file num %i" % i, func="time")
                BleuCalculator.analyse_single(golds, pred, curk, n)
                utils.zlog("", func="time")
        if len(preds) > 1:
            BleuCalculator.analyse_multi(golds, preds, 1, n)
            utils.zlog("", func="time")

# single sentence bleu
def bleu_single(hyp, refs, n=4):
    # hyp: list of tokens; refs: list of list of tokens
    ngrams_ref = [defaultdict(int) for _ in range(n)]
    lengths_ref = []
    # count refs
    for one_ref in refs:
        lengths_ref.append(len(one_ref))
        ngrams_one_ref = BleuCalculator.count_ngrams(one_ref, n)
        for onegram_ref, onegram_one_ref in zip(ngrams_ref, ngrams_one_ref):
            for k in onegram_one_ref:
                onegram_ref[k] = max(onegram_ref[k], onegram_one_ref[k])
    # count hyp
    ngrams_hyp = BleuCalculator.count_ngrams(hyp, n)
    length_hyp = len(hyp)
    crl = BleuCalculator.closest_ref_length(lengths_ref, length_hyp)
    info, info2 = [crl], [length_hyp]
    for _i in range(n):
        info2.append(max(info2[0]-_i, 0))
    for onegram_gold, onegram_one_pred in zip(ngrams_ref, ngrams_hyp):
        cc = 0
        for k in onegram_one_pred:
            cc += min(onegram_gold[k], onegram_one_pred[k])
        info.append(cc)
    one_sb = BleuCalculator.bleu4(info[0], info2[0], info[1:], info2[1:], smoothing=True, report=False)
    return one_sb[0]
