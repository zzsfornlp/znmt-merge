from zl.trainer import Trainer
from zl.search2 import StateStater, extract_states
from zl import data, utils
from zl.model import Model
from .mt_length import get_normer
from .mt_outputter import Outputter
from . import mt_search, mt_eval
from collections import defaultdict, Iterable
import numpy as np
from .mt_par import ParafMap, do_med, MedSegMerger, CovChecker
import pickle

ValidResult = list

class OnceRecorder(object):
    def __init__(self, name, mm=None):
        self.name = name
        self.loss = defaultdict(float)
        self.sents = 1e-6
        self.words = 1e-6
        self.updates = 0
        self.timer = utils.Timer("")
        self._mm = mm

    def record(self, insts, loss, update):
        for k in loss:
            self.loss[k] += loss[k]
        self.sents += len(insts)
        self.words += sum([len(x[0]) for x in insts])     # for src
        self.updates += update

    def reset(self):
        self.loss = self.loss = defaultdict(float)
        self.sents = 1e-5
        self.words = 1e-5
        self.updates = 0
        self.timer = utils.Timer("")
        #
        if self._mm is not None:
            self._mm.stat_clear()

    # const, only reporting, could be called many times
    def state(self):
        one_time = self.timer.get_time()
        loss_per_sentence = "_".join(["%s:%.3f"%(k, self.loss[k]/self.sents) for k in sorted(self.loss.keys())])
        loss_per_word = "_".join(["%s:%.3f"%(k, self.loss[k]/self.words) for k in sorted(self.loss.keys())])
        sent_per_second = float(self.sents) / one_time
        word_per_second = float(self.words) / one_time
        return ("Recoder <%s>, %.3f(time)/%s(updates)/%.1f(sents)/%.1f(words)/%s(sl-loss)/%s(w-loss)/%.3f(s-sec)/%.3f(w-sec)" % (self.name, one_time, self.updates, self.sents, self.words, loss_per_sentence, loss_per_word, sent_per_second, word_per_second))

    def report(self, s=""):
        utils.zlog(s+self.state(), func="info")
        if self._mm is not None:
            self._mm.stat_report()

class MTTrainer(Trainer):
    def __init__(self, opts, model):
        super(MTTrainer, self).__init__(opts, model)

    def _validate_len(self, dev_iter):
        # sqrt error
        count = 0
        loss = 0.
        with utils.Timer(tag="VALID-LEN", print_date=True) as et:
            utils.zlog("With lg as %s." % (self._mm.lg.obtain_params(),))
            for insts in dev_iter.arrange_batches():
                ys = [i[1] for i in insts]
                ylens = np.asarray([len(_y) for _y in ys])
                count += len(ys)
                Model.new_graph()
                self._mm.refresh(False)
                preds = self._mm.predict_length(insts)
                loss += np.sum((preds - ylens) ** 2)
        return - loss / count

    def _validate_ll(self, dev_iter):
        # log likelihood
        one_recorder = self._get_recorder("VALID-LL")
        for insts in dev_iter.arrange_batches():
            loss = self._mm.fb(insts, False, run_name="std2")   # only run fb-mle
            one_recorder.record(insts, loss, 0)
        one_recorder.report()
        # todo(warn) "y" as the key
        return -1 * (one_recorder.loss["y"] / one_recorder.words)

    def _validate_bleu(self, dev_iter):
        # bleu score
        # temp change bsize
        origin_bs = dev_iter.bsize()
        dev_iter.bsize(self.opts["test_batch_size"])
        output_name = self.opts["dev_output"] + ".%s" % self._tp.eidx + ".%s" % self._tp.uidx
        mt_decode("beam", dev_iter, [self._mm], self._mm.target_dict, self.opts, output_name)
        dev_iter.bsize(origin_bs)
        # no restore specifies for the dev set
        s = mt_eval.evaluate(output_name, self.opts["dev"][1], self.opts["eval_metric"], True)
        return s

    def _validate_them(self, dev_iter, metrics):
        validators = {"ll": self._validate_ll, "bleu": self._validate_bleu, "len": self._validate_len}
        r = []
        for m in metrics:
            s = validators[m](dev_iter)
            r.append(float("%.3f" % s))
        return ValidResult(r)

    def _get_recorder(self, name):
        return OnceRecorder(name, self._mm)

    def _fb_once(self, insts):
        return self._mm.fb(insts, True)

# ------------------------------
# decoding

# handling the logging and outputting of the results
class ResultLogger(object):
    def __init__(self, outf, target_dict, opts):
        self.outf = outf
        self.target_dict = target_dict
        self.opts = opts
        # opts
        self.decode_write_once = opts["decode_write_once"]
        self.output_r2l = opts["decode_output_r2l"]
        self.output_kbest = opts["decode_output_kbest"]
        self.decode_dump_hiddens = opts["decode_dump_hiddens"]
        self.decode_dump_sg = opts["decode_dump_sg"]
        #
        if not self.decode_write_once:
            self.tmp_fd_maps = self._create_fdm(self.outf+".tmp")
        else:
            self.tmp_results = []
        self.write_count = 0
        #
        self.num_insts = 0
        self.num_ends = 0
        self.num_mends = 0
        self.ls = [[], []]     # (length, search-size)

    def _add_log(self, rs):
        #
        lengths = [r[0].length for r in rs]
        sizes = [count_states(r[0].sg.get_real_self()) for r in rs]
        self.ls[0] += lengths
        self.ls[1] += sizes
        self.num_insts += len(rs)
        self.num_ends += sum(lengths)
        self.num_mends += sum(sizes)

    def _report_log(self):
        utils.zlog("ResultLogger: %d/%.3f/%.3f/%.3f" % (self.num_insts, self.num_ends/self.num_insts, self.num_mends/self.num_insts, self.num_mends/self.num_ends))
        with utils.zopen("length_sizes.txt", "w") as fd:
            _tmp_idx = 0
            for zl, zs in zip(self.ls[0], self.ls[1]):
                _tmp_idx += 1
                fd.write("%d %d " % (zl, zs))
                if _tmp_idx % 100 == 0:
                    fd.write("\n")

    def _create_fdm(self, outf, mode="w"):
        fdm = {"r":None, "kb":None, "kbs":None, "kbg":None, "hid":None}
        fdm["r"] = utils.zopen(outf, mode)
        if self.output_kbest:
            fdm["kb"] = utils.zopen(outf+".nbest", mode)
            fdm["kbs"] = utils.zopen(outf+".nbests", mode)
            fdm["kbg"] = utils.zopen(outf+".nbestg", mode)
            if self.decode_dump_sg:
                fdm["sg"] = utils.zopen(outf+".sg", mode+"b", None)
            if self.decode_dump_hiddens:
                fdm["hid"] = utils.zopen(outf+".hid", mode+"b", None)
        return fdm

    def _close_fdm(self, fdm):
        for name in fdm:
            fd = fdm[name]
            if fd is not None:
                fd.close()

    def _write_fdm(self, fdm, results):
        ot = Outputter(self.opts)
        if True:
            f = fdm["r"]
            for r in results:
                f.write(ot.format(r, self.target_dict, False, False, self.output_r2l))
            # ot.report()
        if self.output_kbest:
            # todo(warn): specified file names
            if True:
                f = fdm["kb"]
                for r in results:
                    f.write(ot.format(r, self.target_dict, True, False, self.output_r2l))
                # ot.report()
            if True:
                f = fdm["kbs"]
                for r in results:
                    f.write(ot.format(r, self.target_dict, True, True, self.output_r2l))
                # ot.report()
            if True:
                f = fdm["kbg"]
                for i, r in enumerate(results):
                    f.write("# znumber%s\n%s\n" % (i+self.write_count, r[0].sg.get_real_self().show_graph(self.target_dict, False)))
            if self.decode_dump_sg:
                f = fdm["sg"]
                for r in results:
                    pickle.dump(r[0].sg.get_real_self(), f)
            # dump state hiddens
            if self.decode_dump_hiddens:
                f = fdm["hid"]
                for r in results:
                    pickle.dump(extract_states(r[0].sg.get_real_self()), f)
        self.write_count += len(results)

    def add(self, rs):
        self._add_log(rs)
        if not self.decode_write_once:
            self._write_fdm(self.tmp_fd_maps, rs)
        else:
            self.tmp_results += rs

    def _restore_order(self, tracking_list, x):
        if not tracking_list:
            return x
        # todo(warn): BatchArranger.restore_order
        utils.zcheck_matched_length(tracking_list, x, _forced=True)
        ret = [None for _ in x]
        for idx, one in zip(tracking_list, x):
            utils.zcheck_type(ret[idx], type(None), "Wrong tracking list, internal error!!", _forced=True)
            ret[idx] = one
        return ret

    def _rw_one(self, infd, outfd, _ff):
        rlines = [line for line in infd]
        wlines = _ff(rlines)
        for line in wlines:
            outfd.write(line)

    def _rw_multi(self, infd, outfd, _ff):
        records = []
        prev_rec = ""
        for line in infd:
            if len(line.strip()) == 0:
                if len(prev_rec)>0:
                    records.append(prev_rec)
                    prev_rec = ""
            else:
                prev_rec += line
        if len(prev_rec)>0:
            records.append(prev_rec)
        wrecords = _ff(records)
        for rec in wrecords:
            outfd.write(rec+"\n")

    def _rw_pickle(self, infd, outfd, _ff):
        records = []
        try:
            while True:
                one = pickle.load(infd)
                records.append(one)
        except EOFError:
            pass
        wrecords = _ff(records)
        for rec in wrecords:
            pickle.dump(rec, outfd)

    def finish(self, tracking_list):
        if not self.decode_write_once:
            self._close_fdm(self.tmp_fd_maps)
            rfdm = self._create_fdm(self.outf+".tmp", "r")
            wfdm = self._create_fdm(self.outf, "w")
            ff = lambda x: self._restore_order(tracking_list, x)
            # ugly code here
            self._rw_one(rfdm["r"], wfdm["r"], ff)
            if self.output_kbest:
                self._rw_multi(rfdm["kb"], wfdm["kb"], ff)
                self._rw_multi(rfdm["kbs"], wfdm["kbs"], ff)
                self._rw_multi(rfdm["kbg"], wfdm["kbg"], ff)
                if self.decode_dump_sg:
                    self._rw_pickle(rfdm["sg"], wfdm["sg"], ff)
                if self.decode_dump_hiddens:
                    self._rw_pickle(rfdm["hid"], wfdm["hid"], ff)
        else:
            reordered_results = self._restore_order(tracking_list, self.tmp_results)
            fdm = self._create_fdm(self.outf)
            self._write_fdm(fdm, reordered_results)
            self._close_fdm(fdm)
        self._report_log()

# helper: count comb states: similar to extract nbest
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
                combined_list = one.get(_PRUNE_KEY)
                if combined_list is None:
                    combined_list = []
                utils.zcheck_ff_iter(combined_list, lambda x: x.get(_TMP_KEY) is None, "Not scientific pruning states!!")
                # todo(warn): be careful that returning is modifiable
                combined_list = combined_list + [one]
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

# for better printing
def cleanup_states(states_map):
    idx = 0
    states_list = []
    while idx in states_map:
        states_list.append(states_map[idx])
        idx += 1
    #
    assert len(states_list[0])==1
    prev_map = {states_list[0][0].id: 0}
    for one_list in states_list[1:]:
        one_list.sort(key=lambda x: [prev_map[x.pid], x.action_score()])
        for i, one in enumerate(one_list):
            prev_map[one.id] = i
    return states_list

def mt_decode(decode_way, test_iter, mms, target_dict, opts, outf, gold_iter=None):
    reranking = (gold_iter is not None)
    looping = isinstance(test_iter, Iterable)
    if reranking:
        cur_searcher = mt_search.search_rerank
    else:
        cur_searcher = {"greedy":mt_search.search_greedy, "beam":mt_search.search_beam,
                        "sample":mt_search.search_sample, "branch":mt_search.search_branch}[decode_way]
    one_recorder = OnceRecorder("DECODE")
    num_sents = len(test_iter)
    cur_sents = 0.
    sstater = StateStater()
    if opts["decode_extract_paraf"]:
        para_extractor = ParafExtractor(opts, target_dict)
    else:
        para_extractor = DummyParafExtractor(opts)
    # decoding them all
    results = ResultLogger(outf, target_dict, opts)
    tracking_list = None
    prev_point = 0
    # init normer
    for i, _m in enumerate(mms):
        _lg_params = _m.lg.obtain_params()
        utils.zlog("Model[%s] is with lg as %s." % (i, _lg_params,))
    _sigma = np.average([_m.lg.get_real_sigma() for _m in mms], axis=0)
    normer = get_normer(opts["normalize_way"], opts["normalize_alpha"], _sigma)
    # todo: ugly code here
    if looping:
        rs = cur_searcher(mms, test_iter, target_dict, opts, normer, sstater, para_extractor)
        results.add(rs)
        tracking_list = None
    else:
        if reranking:
            for one_tests, one_golds in zip(test_iter.arrange_batches(), gold_iter.arrange_batches()):
                if opts["verbose"] and (cur_sents - prev_point) >= (opts["report_freq"]*test_iter.bsize()):
                    utils.zlog("Reranking process: %.2f%%" % (cur_sents / num_sents * 100))
                    prev_point = cur_sents
                cur_sents += len(one_tests)
                mt_search.search_init()
                rs = cur_searcher(mms, [one_tests, one_golds], target_dict, opts, normer, sstater, para_extractor)
                results.add(rs)
                one_recorder.record(one_tests, {}, 0)
        else:
            for insts in test_iter.arrange_batches():
                if opts["verbose"] and (cur_sents - prev_point) >= (opts["report_freq"]*test_iter.bsize()):
                    utils.zlog("Decoding process: %.2f%%" % (cur_sents / num_sents * 100))
                    prev_point = cur_sents
                cur_sents += len(insts)
                mt_search.search_init()
                # return list(batch) of list(beam) of states
                rs = cur_searcher(mms, insts, target_dict, opts, normer, sstater, para_extractor)
                results.add(rs)
                one_recorder.record(insts, {}, 0)
        one_recorder.report()
        tracking_list = test_iter.get_tracking_list()
        # restore from sorting by length
        # results = test_iter.restore_order(results)
    # output
    sstater.report()
    results.finish(tracking_list)
    para_extractor.save_parafs(outf)
    utils.zlog("COV-LOG: " + CovChecker.report())
    if looping:
        return rs
    else:
        return None

# only for beam_searcher and branch_searcher
class DummyParafExtractor(object):
    def __init__(self, opts):
        self.record = lambda x: None
        self.save_parafs = lambda x: None

class ParafExtractor(object):
    def __init__(self, opts, target_dict):
        self.opts = opts
        #
        self.med_map = ParafMap()
        self.merge_map = ParafMap()
        # med
        self.decode_paraf_nosub = opts["t2_med_nosub"]
        self.med_seg_merger = MedSegMerger(opts, target_dict)
        # merge
        # general

    def record(self, ranked_beam):
        best_end = ranked_beam[0]
        sg = best_end.sg
        inst = sg.src_info
        info = str(inst)
        trg_dict = sg.target_dict
        # first: best vs refs
        # todo(warn): depend on specialized TextInstance
        idx = 1
        best_one = [one.action_code for one in best_end.get_path()]
        while idx < len(inst):
            ref_one = inst[idx]
            idx += 1
            self._extract_med(best_one, ref_one, trg_dict, info)
        # second: extract on merge points
        self._extract_merge(sg, trg_dict, info)

    def _extract_med(self, best_one, ref_one, trg_dict, info):
        tokens_best = best_one
        words_best = [trg_dict.getw(one) for one in tokens_best]
        tokens_ref = ref_one
        words_ref = [trg_dict.getw(one) for one in tokens_ref]
        ori_segs = do_med(tokens_best, tokens_ref, list(reversed(words_best)), list(reversed(words_ref)), not self.decode_paraf_nosub)
        # merging too short matched segs
        segs, _ = self.med_seg_merger.merge(ori_segs)
        # extract the different ones
        ctxs = [words_best, words_ref]
        for ss in segs:
            if not ss[0]:
                self.med_map.add([list(reversed(ss[1])), list(reversed(ss[2]))], ctxs, info)

    def _extract_merge(self, sg, trg_dict, info):
        def _extract_two(one, pruned_one):
            tokens_one = [one.action_code for one in one.get_path()]
            tokens_pruned = [one.action_code for one in pruned_one.get_path()]
            words_one = [trg_dict.getw(one) for one in tokens_one]
            words_pruned = [trg_dict.getw(one) for one in tokens_pruned]
            # simply check two ends
            left_matched_length = 0
            while left_matched_length<len(tokens_one) and left_matched_length<len(tokens_pruned) \
                and tokens_one[left_matched_length] == tokens_pruned[left_matched_length]:
                left_matched_length += 1
            right_idx0, right_idx1 = len(tokens_one)-1, len(tokens_pruned)-1
            while right_idx0>=left_matched_length and right_idx1>=left_matched_length \
                    and tokens_one[right_idx0] == tokens_pruned[right_idx1]:
                right_idx0 -= 1
                right_idx1 -= 1
            # collect
            segs = []
            for words, end in ([words_one, right_idx0], [words_pruned, right_idx1]):
                cur_seg = []
                cur_idx = left_matched_length
                while cur_idx <= end:
                    cur_seg.append(words[cur_idx])
                    cur_idx += 1
                segs.append(cur_seg)
            self.merge_map.add(segs, [words_one, words_pruned], info)
        #
        # check all states for merged points
        them = sg.bfs()
        for s0, s1 in them:
            for one in s0+s1:
                prunings = one.get("PRUNING_LIST")
                if prunings is not None:
                    for pruned_one in prunings:
                        _extract_two(one, pruned_one)

    def save_parafs(self, outf_prefix):
        with utils.zopen(outf_prefix+".med.json", "w") as fd:
            self.med_map.save(fd)
        with utils.zopen(outf_prefix+".merge.json", "w") as fd:
            self.merge_map.save(fd)
