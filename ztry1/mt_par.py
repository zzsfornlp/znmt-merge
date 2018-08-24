# paraf: about med & alignments
import json, numpy
# from scipy.spatial import distance

VNAME_ATTW = "attw"
VNAME_CREC = "crec"

class CovChecker(object):
    # --- statistic summaries ---
    all_cov_dists = []
    all_hid_dists = []
    all_med_dists = []

    @staticmethod
    def clear():
        CovChecker.all_cov_dists = []
        CovChecker.all_hid_dists = []
        CovChecker.all_med_dists = []

    @staticmethod
    def report():
        return "c:%.3f*%d,h:%.3f*%d,m%.3f*%d" % \
               (numpy.average(CovChecker.all_cov_dists), len(CovChecker.all_cov_dists),
                numpy.average(CovChecker.all_hid_dists), len(CovChecker.all_hid_dists),
                numpy.average(CovChecker.all_med_dists), len(CovChecker.all_med_dists))
    # --- statistic summaries ---

    def __init__(self, opts):
        self.opts = opts
        self.cov_record_mode = opts["cov_record_mode"]
        self.cov_l1_thresh = opts["cov_l1_thresh"]
        self.cov_upper_bound = opts["cov_upper_bound"]
        self.cov_average = opts["cov_average"]
        #
        self.hid_sim_metric = opts["hid_sim_metric"]
        self.hid_sim_thresh = opts["hid_sim_thresh"]
        #
        self.merge_diff_metric = opts["merge_diff_metric"]
        self.merge_diff_thresh = opts["merge_diff_thresh"]
        #
        self._update_f = {"none": lambda prev,cur:prev,
                          "max": CovChecker.update_max,
                          "sum": lambda prev,cur:prev+cur}[self.cov_record_mode]
        # distance as [0,1]
        self._dist_f_collections = {"none": lambda x,y: 1.0,
                        "cos": lambda x,y: (1.0 - numpy.dot(x, y)/(numpy.linalg.norm(x)*numpy.linalg.norm(y)))/2,
                        "c1": lambda x,y: numpy.sum(numpy.abs(x-y)/(numpy.sum(numpy.abs(x))+numpy.sum(numpy.abs(y)))),
                        "c2": lambda x,y: numpy.sqrt(numpy.sum(numpy.square(x-y))/(numpy.sum(numpy.abs(x))+numpy.sum(numpy.abs(y))))
                        }
        self._dist_f = self._dist_f_collections[self.hid_sim_metric]

    def cov_ok(self, a, b):
        ret = True
        if self.cov_record_mode != "none":
            cov_a = self._get_cov_record(a)
            cov_b = self._get_cov_record(b)
            cov_a = numpy.minimum(cov_a, self.cov_upper_bound)
            cov_b = numpy.minimum(cov_b, self.cov_upper_bound)
            if self.cov_average:
                cov_a /= a.length
                cov_b /= b.length
            d0 = numpy.sum(numpy.abs(cov_a-cov_b))
            CovChecker.all_cov_dists.append(d0)
            if d0 > self.cov_l1_thresh:
                ret = False
        if self.hid_sim_metric != "none":
            hid_a = a.prev.get_hidv()[0]
            hid_b = b.prev.get_hidv()[0]
            d1 = self._dist_f(hid_a, hid_b)
            CovChecker.all_hid_dists.append(d1)
            if d1 > self.hid_sim_thresh:
                ret = False
            # ret = ret and (d <= self.hid_sim_thresh)
        if self.merge_diff_metric == "med":
            seq_a = [x.action_code for x in a.get_path()]
            seq_b = [x.action_code for x in b.get_path()]
            d2 = do_med(seq_a, seq_b, None, None, True, True)
            CovChecker.all_med_dists.append(d2)
            if d2 > self.merge_diff_thresh:
                ret = False
        return ret

    #
    @staticmethod
    def update_max(prev, cur):
        adding = numpy.zeros(cur.shape)
        max_idx = cur.argmax()
        adding[max_idx] += 1
        ret = prev + adding
        return ret

    def _get_cov_record(self, one):
        ret = one.get(VNAME_CREC)
        if ret is None:
            if one.is_start():
                ret = 0
            else:
                prev_rec = self._get_cov_record(one.prev)
                cur = one.get(VNAME_ATTW)
                ret = self._update_f(prev_rec, cur)
            one.set(VNAME_CREC, ret)
        return ret


class MedSegMerger(object):
    # eliminate suspicious matched seq (maybe MedSegJudger is a better name)
    def __init__(self, opts, target_dict):
        self.target_dict = target_dict    # already sorted
        self.seg_minlen = opts["t2_err_seg_minlen"]
        self.seg_freq_token = opts["t2_err_seg_freq_token"]
        self.seg_extends = opts["t2_err_seg_extend_range"]
        self.cov_checker = CovChecker(opts)

    def get_tok(self, x):
        if hasattr(x, 'action_code'):
            return x.action_code
        elif type(x) == int:
            return x
        elif type(x) == str:
            return self.target_dict[x]
        else:
            return None

    def is_freq(self, x):
        if type(x) == int:
            return x<=self.seg_freq_token
        else:
            return self.get_tok(x)<=self.seg_freq_token

    # return (possibly merged segs, #false_match)
    def merge(self, ori_segs):
        cur_falsematch = {"len":0,"cov":0}
        segs = []
        # first scan of length
        for ss in ori_segs:
            real_flag = ss[0]
            length_nofreq = sum(0 if self.is_freq(tok) else 1 for tok in ss[1])
            if real_flag:
                cancelling = False
                if length_nofreq < self.seg_minlen:
                    cancelling = True
                    cur_falsematch["len"] += len(ss[1])
                elif not self.cov_checker.cov_ok(ss[1][0], ss[2][0]):   # #0 is the last one because of reversing
                    cancelling = True
                    cur_falsematch["cov"] += len(ss[1])
                if cancelling:
                    real_flag = False
            if len(segs)>0 and segs[-1][0] == real_flag:
                # merge into one
                segs[-1] = (real_flag, segs[-1][1]+ss[1], segs[-1][2]+ss[2])
            else:
                segs.append((real_flag, ss[1], ss[2]))
        # second scan of reorder
        ori_segs2 = segs
        segs2 = []
        length_segs2 = len(ori_segs2)
        # todo(warn): remember segs2 are reversed
        for idx in range(length_segs2):
            ss = ori_segs2[idx]
            real_flag = ss[0]
            if real_flag:
                # (past_pred - past_ref) ^ future_ref
                past_ref_rev_states = ori_segs2[idx+1][2] if idx+1<length_segs2 else []
                past_ref_rev_toks_set = set([self.get_tok(x) for x in past_ref_rev_states])
                past_pred_rev_states = ori_segs2[idx+1][1][:self.seg_extends] if idx+1<length_segs2 else []
                past_pred_rev_toks = []
                for x in past_pred_rev_states:
                    c = self.get_tok(x)
                    if x not in past_ref_rev_toks_set:
                        past_pred_rev_toks.append(c)
                # todo(warn): simply 1-gram match
                tmp_tok_set = set(past_pred_rev_toks)
                future_ref_rev_states = segs2[-1][2][-self.seg_extends:] if len(segs2)>0 else []
                future_ref_rev_toks = [self.get_tok(x) for x in future_ref_rev_states]
                tmp_matched_count = sum(1 if (tok in tmp_tok_set) and not self.is_freq(tok) else 0 for tok in future_ref_rev_toks)
                if tmp_matched_count>0:
                    cur_falsematch += len(ss[1])
                    real_flag = False
            if len(segs2)>0 and segs2[-1][0] == real_flag:
                # merge into one
                segs2[-1] = (real_flag, segs2[-1][1]+ss[1], segs2[-1][2]+ss[2])
            else:
                segs2.append((real_flag, ss[1], ss[2]))
        return segs2, cur_falsematch

# return rev-list of (if-matched, rev-beam, rev-gold)
def do_med(tokens_beam, tokens_gold, rev_list_beam, rev_list_gold, if_sub, return_dist=False):
    EDIT_DEL, EDIT_ADD, EDIT_SUB, EDIT_MATCH = 0,1,2,3
    length_beam, length_gold = len(tokens_beam), len(tokens_gold)
    max_edits = length_beam + length_gold
    # tables: T[BEAM_LEN+1][GOLD_LEN+1]
    tab_dis, tab_act = [[n for n in range(length_gold+1)]], [[EDIT_MATCH] + [EDIT_ADD]*(length_gold)]
    # dp -- n*m loop
    for i, tok_beam in enumerate(tokens_beam):
        last_dis = tab_dis[-1]
        one_dis, one_act = [i+1], [EDIT_DEL]
        tab_dis.append(one_dis)
        tab_act.append(one_act)
        for j, tok_gold in enumerate(tokens_gold):
            v_ij, v_ipj, v_ijp = last_dis[j], one_dis[j], last_dis[j+1]
            if tok_beam == tok_gold:
                this_dis = v_ij
                this_act = EDIT_MATCH
            else:
                # using specific orders for the operations
                # sub
                if if_sub:
                    this_dis = v_ij+1
                    this_act = EDIT_SUB
                else:
                    this_dis = max_edits
                    this_act = None
                # add
                if this_dis > v_ipj+1:
                    this_dis = v_ipj+1
                    this_act = EDIT_ADD
                # del
                if this_dis > v_ijp+1:
                    this_dis = v_ijp+1
                    this_act = EDIT_DEL
            #
            one_dis.append(this_dis)
            one_act.append(this_act)
    if return_dist:
        return tab_dis[-1][-1]
    # back-tracking and get segs
    def _add_if_nonempty(sgs, one):
        if len(one[1])>0 or len(one[2])>0:
            sgs.append(one)
    segs = []
    cur_one = (True, [], [])
    idx_beam, idx_gold = 0, 0
    while idx_beam<length_beam or idx_gold<length_gold:
        _i, _j = length_beam-idx_beam, length_gold-idx_gold
        act = tab_act[_i][_j]
        if act == EDIT_MATCH:
            if not cur_one[0]:
                _add_if_nonempty(segs, cur_one)
                cur_one = (True, [], [])
            cur_one[1].append(rev_list_beam[idx_beam])
            idx_beam += 1
            cur_one[2].append(rev_list_gold[idx_gold])
            idx_gold += 1
        else:
            if cur_one[0]:
                _add_if_nonempty(segs, cur_one)
                cur_one = (False, [], [])
            if act == EDIT_SUB:
                cur_one[1].append(rev_list_beam[idx_beam])
                idx_beam += 1
                cur_one[2].append(rev_list_gold[idx_gold])
                idx_gold += 1
            elif act == EDIT_ADD:
                cur_one[2].append(rev_list_gold[idx_gold])
                idx_gold += 1
            elif act == EDIT_DEL:
                cur_one[1].append(rev_list_beam[idx_beam])
                idx_beam += 1
            else:
                raise RuntimeError("Unlegal action for med.")
    _add_if_nonempty(segs, cur_one)
    return segs

# ====================
# para instance

# {"m1": [], "m2": [], "ctx1": [], ctx2": []}
# class ParaInst(object):
#     pass

class ParafMap(object):
    def __init__(self):
        self.maps = {}          # key -> {kall -> (num, [list of inst])}
        self.key_maps = {}   # key -> {(num, [list of inst])}
        self.instances = []

    def _key(self, one_list):
        return " ".join([str(s) for s in one_list])

    def _get_keys(self, inst):
        # return [keys], all_key
        k1 = self._key(inst["m1"])
        k2 = self._key(inst["m2"])
        kall = k1+"|||"+k2
        return [k1,k2], kall

    def _norm_inst(self, inst):
        ms = [inst["m1"], inst["m2"]]
        ctxs = [inst["ctx1"], inst["ctx2"]]
        info = inst["info"]
        if self._key(ms[0]) >= self._key(ms[1]):
            inst = {"m1":ms[1], "m2":ms[0], "ctx1":ctxs[1], "ctx2":ctxs[0], "info":info, "reorder":True}
        return inst

    # True if pruned
    @staticmethod
    def filter_inst(inst, min_len=0, max_len=50, max_abs=10, no_unk=False):
        len1, len2 = len(inst["m1"]), len(inst["m2"])
        left, right = min(len1,len2), max(len1,len2)
        if left<min_len:
            return True
        if right>max_len:
            return True
        if right-left>max_abs:
            return True
        if no_unk:
            for one_list in [inst["m1"], inst["m2"]]:
                if any(one=="<unk>" for one in one_list):
                    return True
        return False

    def add_inst(self, inst):
        inst = self._norm_inst(inst)
        ks, kall = self._get_keys(inst)
        self.instances.append(inst)
        # add maps
        for i, k in enumerate(ks):
            if k not in self.maps:
                self.maps[k] = {}
            key_map = self.maps[k]
            if kall not in key_map:
                key_map[kall] = [1, [inst]]
            else:
                key_map[kall][0] += 1
                key_map[kall][1].append(inst)
        # add key_maps
        if kall not in self.key_maps:
            self.key_maps[kall] = [1, [inst]]
        else:
            self.key_maps[kall][0] += 1
            self.key_maps[kall][1].append(inst)

    def add(self, ms, ctxs, info):
        inst = {"m1":ms[0], "m2":ms[1], "ctx1":ctxs[0], "ctx2":ctxs[1], "info":info, "reorder":False}
        self.add_inst(inst)

    def save(self, fd):
        ParafMap.write_to(self, fd)

    @staticmethod
    def write_to(pm, fd):
        for s in pm.instances:
            fd.write(json.dumps(s)+"\n")

    @staticmethod
    def read_from(fd, inst_filter=None):
        instances = [json.loads(line) for line in fd]
        one = ParafMap.create_from(instances, inst_filter)
        return one

    @staticmethod
    def create_from(instances, inst_filter):
        if inst_filter is None:
            inst_filter = lambda x: False
        one = ParafMap()
        for s in instances:
            if not inst_filter(s):
                one.add_inst(s)
        return one

    # query with conditions
    def query(self, ss):
        key = self._key(ss.split())
        if key in self.maps:
            r = self.maps[key]
            ret = []
            for key_all in r:
                ret.append((key_all, r[key_all][0], r[key_all][1]))
            ret.sort(key=lambda x:x[1])
            return ret
        else:
            return []

    def filter(self, inst_filter):
        one = ParafMap.create_from(self.instances, inst_filter)
        return one

    def info(self, topn=10):
        # describe self
        num_inst = len(self.instances)
        num_pairs = len(self.key_maps)
        all_pairs = list(self.key_maps.keys())
        all_pairs.sort(key=lambda k:self.key_maps[k][0], reverse=True)
        s = "#instances=%d, #pairs=%d, and top-%d are:\n" % (num_inst, num_pairs, topn)
        for i in range(topn):
            k = all_pairs[i]
            s0 = "%04d: %s #%d" % (i, k, self.key_maps[k][0])
            s += s0 + "\n"
        return s

# ============================================
if __name__ == '__main__':
    import argparse, sys, traceback

    # specs about filters
    def get_filter_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--minlen', type=int, default=0)
        parser.add_argument('--maxlen', type=int, default=50)
        parser.add_argument('--maxabs', type=int, default=10)
        parser.add_argument('--no_unk', action='store_true')
        parser.add_argument('--with_filter', action='store_true')
        return parser

    class Driver(object):
        def __init__(self):
            self.maps = []
            self.maps_src = []
            self.cmds = []
            self.filter_parser = get_filter_parser()
            self.exit = False

        def printing(self, s, end='\n', err=False):
            fff = sys.stderr if err else sys.stdout
            print(s, end=end, file=fff, flush=True)

        def record(self, cmd, descr):
            self.printing(descr, err=True)
            self.cmds.append((cmd, descr))

        # operations
        def op_load(self, file, cmd):
            with open(file) as fd:
                mm = ParafMap.read_from(fd)
                descr = "-- Load map %d from %s" % (len(self.maps), file, )
                self.maps.append(mm)
                self.maps_src.append(len(self.cmds))    # created from which cmd
                self.record(cmd, descr)

        def op_save(self, num, file, cmd):
            mm = self.maps[num]
            with open(file, 'w') as fd:
                mm.save(fd)
                descr = "-- Save map %d to %s." % (num, file)
                self.record(cmd, descr)

        def op_exit(self, cmd):
            descr = "!! Bye."
            self.record(cmd, descr)
            self.exit = True

        def op_filter(self, num, inst_filter, cmd):
            mm0 = self.maps[num]
            mm1 = mm0.filter(inst_filter)
            descr = "-- Filter map %d to %d." % (num, len(self.maps))
            self.maps.append(mm1)
            self.maps_src.append(len(self.cmds))
            self.record(cmd, descr)

        def op_list(self, cmd):
            descr = "-- List all maps."
            for i, cmd_idx in enumerate(self.maps_src):
                self.printing("** Map %d: %s" % (i, self.cmds[cmd_idx]))
            self.record(cmd, descr)

        def op_query(self, num, s, verbose, cmd):
            MAX_ITEMS = 10
            descr = "-- Query map %d with %s." % (num, s)
            mm = self.maps[num]
            rets = mm.query(s)
            self.printing("== With answer of %d:" % len(rets))
            for i, r in enumerate(rets):
                self.printing("** Answer #%d: %s #%s" % (i, r[0], r[1]))
                if verbose:
                    for one in r[-1][:MAX_ITEMS]:
                        self.printing("   Eg -> %s" % one)
            self.record(cmd, descr)

        def op_info(self, num, topn, cmd):
            descr = "-- Info map %d with %d." % (num, topn)
            mm = self.maps[num]
            info = mm.info(topn)
            self.printing("== info: %s" % info)
            self.record(cmd, descr)

        # driver
        def once(self, line):
            if len(line) == 0:
                self.op_exit(line)
            fields = line.split()
            c = fields[0]
            if c=="load":
                file_name = fields[1]
                self.op_load(file_name, line)
            elif c=="save":
                num, file_name = int(fields[1]), fields[2]
                self.op_save(num, file_name, line)
            elif c=="exit":
                self.op_exit(line)
            elif c=="filter":
                num = int(fields[1])
                args = self.filter_parser.parse_args(fields[2:])
                inst_filter = None
                if args.with_filter:
                    inst_filter = lambda x: ParafMap.filter_inst(x, min_len=args.minlen, max_len=args.maxlen, max_abs=args.maxabs, no_unk=args.no_unk)
                self.op_filter(num, inst_filter, line)
            elif c=="list":
                self.op_list(line)
            elif c=="query" or c=="vquery":
                num = int(fields[1])
                ss = " ".join(fields[2:])
                self.op_query(num, ss, c=="vquery", line)
            elif c=="info":
                num = int(fields[1])
                topn = int(fields[2])
                self.op_info(num, topn, line)
            else:
                raise NotImplementedError("Unkown command %s in %s" % (c, line))

        def loop(self):
            if len(sys.argv) > 1:
                # reload instructions
                self.printing("Load instructions from %s." % sys.argv[1])
                pre_cmds = []
                with open(sys.argv[1]) as f:
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            self.printing("-> %s" % line)
                        pre_cmds.append(line)
                for line in pre_cmds:
                    self.once(line)
            while not self.exit:
                try:
                    self.printing(">> ", end="")
                    line = sys.stdin.readline()
                    self.once(line)
                except:
                    self.printing(traceback.format_exc(), err=True)

    # running
    x = Driver()
    x.loop()

# example:
# load output.007.med.json
# load output.007.merge.json
# filter 0 --minlen 1 --maxlen 5 --maxabs 2 --no_unk --with_filter
# filter 1 --minlen 1 --maxlen 5 --maxabs 2 --no_unk --with_filter
# save 2 test.med.json
# save 3 test.merge.json
# query 0 of course
# query 1 am going to
# load test.json
# -> queries: of course, want, will, would like

# real example:
# python3 ../znmt/ztry1/mt_par.py < med.op |& tee med.log &
# python3 ../znmt/ztry1/mt_par.py < merge.op |& tee merge.log &
# load dt.ZZNAME.json
# filter 0 --minlen 1 --maxlen 5 --no_unk --with_filter
# filter 0 --minlen 1 --maxlen 5 --maxabs 2 --no_unk --with_filter
# filter 0 --minlen 1 --maxlen 10 --no_unk --with_filter
# filter 0 --minlen 1 --maxlen 10 --maxabs 2 --no_unk --with_filter
# filter 0 --minlen 1 --maxlen 10 --maxabs 5 --no_unk --with_filter
# filter 0 --minlen 1 --maxlen 15 --no_unk --with_filter
# filter 0 --minlen 1 --maxlen 15 --maxabs 5 --no_unk --with_filter
# filter 0 --minlen 5 --maxlen 10 --no_unk --with_filter
# save 1 ZZNAME.1.json
# save 2 ZZNAME.2.json
# save 3 ZZNAME.3.json
# save 4 ZZNAME.4.json
# save 5 ZZNAME.5.json
# save 6 ZZNAME.6.json
# save 7 ZZNAME.7.json
# save 8 ZZNAME.8.json
# sed 's/ZZNAME/med/g' < tmp.op > med.op
# sed 's/ZZNAME/merge/g' < tmp.op > merge.op
