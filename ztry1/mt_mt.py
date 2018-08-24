# focusing on speed

from zl.model import Model
from zl import utils, data
from .mt_length import MTLengther, LinearGaussain
import numpy as np
from . import mt_layers as layers
from zl.search import State, SearchGraph, Action
from .mt_search import Pruner as SPruner
from .mt_extern import RAML, ScheduledSamplerSetter
from .mt_par import do_med, MedSegMerger
from .mt_rerank_analysis import bleu_single
from collections import defaultdict

# consts
from zl.backends.common import ResultTopk
#
IDX_DIM = ResultTopk.IDX_DIM
VAL_DIM = ResultTopk.VAL_DIM
# specific running options
VNAME_CACHE = "CC"      # (cache-dict, batch-idx)
VNAME_GOLD = "GD"       # BOOL: whether it is a gold state
VNAME_ATT = "AT"        # State: attach to which gold state if possible
VNAME_ATT_BASE = "AB"   # int: baselength for attaching (last attaching length)
VNAME_SYNCLEN = "SY"    # int: sync length
VNAME_GISTAT = "GI"     # gold-interf mode: means the ends of updating frags
VNAME_ATTW = "attw"     # attention-weights
#
PADDING_STATE = None
PADDING_ACTION = 0

# herlpers
# data helpers #
def prepare_data(ys, dict, fix_len=0):
    # input: list of list of index (bsize, step),
    # output: padded input (step, bsize), masks (1 for real words, 0 for paddings, None for all 1)
    bsize, steps = len(ys), max([len(i) for i in ys])
    if fix_len > 0:
        steps = fix_len
    y = [[] for _ in range(steps)]
    ym = [None for _ in range(steps)]
    lens = [len(i) for i in ys]
    eos, padding = dict.eos, dict.pad
    for s in range(steps):
        _one_ym = []
        _need_ym = False
        for b in range(bsize):
            if s<lens[b]:
                y[s].append(ys[b][s])
                _one_ym.append(1.)
            else:
                y[s].append(padding)
                _one_ym.append(0.)
                _need_ym = True
        if _need_ym:
            ym[s] = _one_ym
    # check last step
    for b in range(bsize):
        if y[-1][b] not in [eos, padding]:
            y[-1][b] = eos
    return y, ym

def prepare_y_step(ys, i):
    _need_mask = False
    ystep = []
    _mask = []
    for _y in ys:
        if i<len(_y):
            ystep.append(_y[i])
            _mask.append(1.)
        else:
            ystep.append(0)
            _mask.append(0)
            _need_mask = True
    if _need_mask:
        mask_expr = layers.BK.inputVector(_mask)
        mask_expr = layers.BK.reshape(mask_expr, (1, ), len(ys))
    else:
        mask_expr = None
    return ystep, mask_expr

# An typical example of a model, fixed architecture
# single s2s: one input(no factors), one output
# !! stateless, states & caches are managed by the Scorer
class s2sModel(Model):
    def __init__(self, opts, source_dict, target_dict, length_info):
        super(s2sModel, self).__init__()
        # helpers
        self.prepare_y_step = prepare_y_step
        #
        self.opts = opts
        self.source_dict = source_dict
        self.target_dict = target_dict
        # build the layers
        # embeddings
        self.embed_src = layers.Embedding(self.model, len(source_dict), opts["dim_word"], dropout_wordceil=source_dict.get_wordceil())
        self.embed_trg = layers.Embedding(self.model, len(target_dict), opts["dim_word"], dropout_wordceil=target_dict.get_wordceil())
        # enc-dec
        self.enc = layers.Encoder(self.model, opts["dim_word"], opts["hidden_enc"], opts["enc_depth"], opts["rnn_type"])
        self.dec_ngram_n = opts["dec_ngram_n"]
        if opts["dec_type"] == "ngram":
            self.dec = layers.NgramDecoder(self.model, opts["dim_word"], opts["hidden_dec"], opts["dec_depth"], 2*opts["hidden_enc"],
                    opts["hidden_att"], opts["att_type"], opts["rnn_type"], opts["summ_type"], self.dec_ngram_n)
        else:
            self.dec = layers.NematusDecoder(self.model, opts["dim_word"], opts["hidden_dec"], opts["dec_depth"], 2*opts["hidden_enc"],
                        opts["hidden_att"], opts["att_type"], opts["rnn_type"], opts["summ_type"])
        # outputs
        self.out0 = layers.Linear(self.model, 2*opts["hidden_enc"]+opts["hidden_dec"]+opts["dim_word"], opts["hidden_out"])
        self.out1 = layers.Linear(self.model, opts["hidden_out"], len(target_dict), act="linear")
        #
        # computation values
        # What is in the cache: S,V,summ/ ctx,att,/ out_s,results
        self.names_bv = {"hid"}
        self.names_bi = {"S", "V"}
        self.names_ig = {"ctx", "att", "out_s", "results", "summ"}
        # length
        self.scaler = MTLengther.get_scaler_f(opts["train_scale_way"], opts["train_scale"])  # for training
        self.lg = LinearGaussain(self.model, 2*opts["hidden_enc"], opts["train_len_xadd"], opts["train_len_xback"], length_info)
        utils.zlog("End of creating Model.")
        # !! advanced options (enabled by MTTrainer)
        self.is_fitting_length = False      # whether adding length loss for training
        self.len_lambda = opts["train_len_lambda"]
        # other model properties
        if opts["train_r2l"]:
            self.set_prop("r2l", True)
        if opts["no_model_softmax"]:
            self.set_prop("no_model_softmax", True)
        self.fb_fs = {"std2":self.fb_standard2_, "beam":self.fb_beam_, "branch":self.fb_branch_}
        self.fber_ = self.fb_fs[opts["train_mode"]]
        self.losser_ = {"mle":self._mle_loss_step, "mlev":self._mlev_loss_step, "hinge_max":self._hinge_max_loss_step, "hinge_avg0":self._hinge_avg0_loss_step, "hinge_avg":self._hinge_avg_loss_step, "hinge_sum":self._hinge_sum_loss_step}[opts["train_local_loss"]]
        self.beam_losser_ = BeamLossBuilder(opts, self.target_dict)
        self.margin_ = opts["train_margin"]
        utils.zlog("For the training process %s, using %s; loss is %s, using %s; margin is %s"
                   % (opts["train_mode"], self.fber_, opts["train_local_loss"], self.losser_, self.margin_))
        #
        self.penalize_eos = opts["penalize_eos"]
        self.penalize_list = self.target_dict.get_ending_tokens()
        #
        self.ss_rate = 1.0  # scheduled sampling rate
        self.ss_size = opts["ss_size"]
        self.sser = ScheduledSamplerSetter(opts)

    def repeat(self, c, bs, times, names):
        new_c = {}
        orders = [i//times for i in range(bs*times)]
        for n in names:
            new_c[n] = layers.BK.rearrange_cache(c[n], orders)
        return new_c

    def rerange(self, c, bv_orders, bi_orders):
        new_c = {}
        for names, orders in ((self.names_bv, bv_orders), (self.names_bi, bi_orders)):
            if orders is not None:
                for n in names:
                    new_c[n] = layers.BK.rearrange_cache(c[n], orders)
            else:
                for n in names:
                    new_c[n] = c[n]
        return new_c

    def recombine(self, clist, idxlist, names=None):
        new_c = {}
        if names is None:
            them = (self.names_bv, self.names_bi)
        else:
            them = (names,)
        for names in them:
            for n in names:
                new_c[n] = layers.BK.recombine_cache([_c[n] for _c in clist], idxlist)
        return new_c

    def refresh(self, training):
        def _gd(drop):  # get dropout
            return drop if (training or self.opts["drop_test"]) else 0.
        opts = self.opts
        self.embed_src.refresh({"hdrop":_gd(opts["drop_embedding"]), "idrop":_gd(opts["idrop_embedding"])})
        self.embed_trg.refresh({"hdrop":_gd(opts["drop_embedding"]), "idrop":_gd(opts["idrop_embedding"])})
        self.enc.refresh({"idrop":_gd(opts["idrop_enc"]), "gdrop":_gd(opts["gdrop_enc"])})
        self.dec.refresh({"idrop":_gd(opts["idrop_dec"]), "gdrop":_gd(opts["gdrop_dec"]), "hdrop":_gd(opts["drop_hidden"])})
        self.out0.refresh({"hdrop":_gd(opts["drop_hidden"])})
        self.out1.refresh({})
        self.lg.refresh({"idrop":_gd(opts["drop_hidden"])})

    # ----------
    def update_schedule(self, uidx):
        # todo, change mode while training (before #num updates)
        # fitting len
        if not self.is_fitting_length and uidx>=self.opts["train_len_uidx"]:
            self.is_fitting_length = True
            utils.zlog("(Advanced-fitting) Model is starting to fit length.")
        # sser
        self.ss_rate = self.sser.get_rate(uidx)

    def stat_report(self):
        if self.opts["train_mode"] == "beam":
            utils.zlog(self.beam_losser_.report(), func="details")

    def stat_clear(self):
        self.beam_losser_.refresh(True)
    # ----------

    # helper routines #
    def get_embeddings_step(self, tokens, embed):
        # tokens: list of int or one int, embed: Embedding => one expression (batched)
        return embed(tokens)

    def get_start_yembs(self, bsize):
        bos = self.target_dict.bos
        return self.get_embeddings_step([bos for _ in range(bsize)], self.embed_trg)

    def get_scores(self, at, hi, ye):
        real_hi = hi[-1]["H"]
        output_concat = layers.BK.concatenate([at, real_hi, ye])
        output_hidden = self.out0(output_concat)
        output_score = self.out1(output_hidden)
        return output_score, output_hidden

    def encode(self, xx, xm):
        # -- encode xs, return list of encoding vectors
        # xx, xm = self.prepare_data(xs) # prepare at the outside
        x_embed = [self.get_embeddings_step(s, self.embed_src) for s in xx]
        x_encodes = self.enc(x_embed, xm)
        return x_encodes

    def decode_start(self, x_encodes):
        # start the first step of decoding
        return self.dec.start_one(x_encodes)

    def decode_step(self, x_encodes, inputs, caches, prev_embeds):
        # feed one step
        return self.dec.feed_one(x_encodes, inputs, caches, prev_embeds)

    # main routines #
    def start(self, xs, repeat_time=1, softmax=True):
        # encode
        bsize = len(xs)
        xx, xm = prepare_data(xs, self.source_dict)
        x_encodes = self.encode(xx, xm)
        x_encodes = [layers.BK.batch_repeat(one, repeat_time) for one in x_encodes]
        # init decode
        cache = self.decode_start(x_encodes)
        start_embeds = self.get_start_yembs(bsize*repeat_time)
        output_score, output_hidden = self.get_scores(cache["ctx"], cache["hid"], start_embeds)
        if softmax and not self.get_prop("no_model_softmax"):
            results = layers.BK.softmax(output_score)
        else:
            results = output_score
        # return
        cache["out_s"] = output_score
        cache["results"] = results
        return cache

    def step(self, prev_val, inputs, cur_states=None, softmax=True):
        x_encodes, hiddens = None, prev_val["hid"]
        next_embeds = self.get_embeddings_step(inputs, self.embed_trg)
        # prepare prev_embeds
        if self.opts["dec_type"] == "ngram":
            bos = self.target_dict.bos
            prev_tokens = [s.sig_ngram_tlist(self.dec_ngram_n, bos) for s in cur_states]
            prev_embeds = [self.get_embeddings_step([p[step] for p in prev_tokens], self.embed_trg) for step in range(self.dec_ngram_n)]
        else:
            prev_embeds = None
        cache = self.decode_step(x_encodes, next_embeds, prev_val, prev_embeds)
        output_score, output_hidden = self.get_scores(cache["ctx"], cache["hid"], next_embeds)
        if softmax and not self.get_prop("no_model_softmax"):
            results = layers.BK.softmax(output_score)
        else:
            results = output_score
        # return
        cache["out_s"] = output_score
        cache["results"] = results
        return cache

    def predict_length(self, insts, cc=None):
        # todo(warn): already inited graph
        # return real lengths
        xs = [i[0] for i in insts]
        xlens = [len(_x) for _x in xs]
        if cc is None:
            cc = self.start(xs, softmax=False)
        pred_lens = self.lg.calculate(cc["summ"], xlens)
        ret = np.asarray(layers.BK.get_value_vec(pred_lens))
        ret = LinearGaussain.back_len(ret)
        return ret

    # for the results: input is pairs(K) of (idx, val)
    def explain_result_topkp(self, pairs):
        to_log = not self.get_prop("no_model_softmax")
        to_penalize = self.penalize_eos > 0.
        ret_pairs = []
        for p in pairs:
            if to_log:
                p[VAL_DIM] = np.log(p[VAL_DIM])
            if to_penalize:
                if p[IDX_DIM] in self.penalize_list:
                    p[VAL_DIM] -= self.penalize_eos
            if p[IDX_DIM] != self.target_dict.err:
                ret_pairs.append(p)
        # re-sorting
        ret_pairs.sort(key=lambda p: p[VAL_DIM], reverse=True)
        return ret_pairs

    # =============================
    # training
    def fb(self, insts, training, ret_value="loss", new_graph=True, run_name=None):
        if new_graph:
            Model.new_graph()
            self.refresh(training)
        # extras.RAML
        if self.opts["raml_samples"] > 0:
            insts = RAML.modify_data(insts, self.target_dict, self.opts)
        #
        if run_name is None:
            r = self.fber_(insts, training, ret_value)
        else:
            r = self.fb_fs[run_name](insts, training, ret_value)
        return r

    # helpers
    def prepare_xy_(self, insts):
        xs = [i[0] for i in insts]
        if self.get_prop("r2l"):
            # right to left modeling, be careful about eos
            ys = [list(reversed(i[1][:-1]))+[i[1][-1]] for i in insts]
        else:
            ys = [i[1] for i in insts]
        return xs, ys

    # specific forward/backward runs
    def fb_standard2_(self, insts, training, ret_value="loss"):
        # please don't ask me where is standard1 ...
        # similar to standard1, but record states and no training for lengths
        xs, ys = self.prepare_xy_(insts)
        bsize = len(xs)
        opens = [State(sg=SearchGraph(target_dict=self.target_dict)) for _ in range(bsize)]     # with only one init state
        # xlens = [len(_x) for _x in xs]
        ylens = [len(_y) for _y in ys]
        cur_maxlen = max(ylens)
        losses = []
        caches = []
        yprev = None
        for i in range(cur_maxlen):
            # forward
            ystep, mask_expr = prepare_y_step(ys, i)
            if i==0:
                cc = self.start(xs, softmax=False)
            else:
                cc = self.step(caches[-1], yprev, opens, softmax=False)
            caches.append(cc)
            # build loss
            scores_exprs = cc["out_s"]
            loss = self.losser_(scores_exprs, ystep, mask_expr)
            if self.scaler is not None:
                len_scales = [self.scaler(len(_y)) for _y in ys]
                np.asarray(len_scales).reshape((1, -1))
                len_scalue_e = layers.BK.inputTensor(len_scales, True)
                loss = loss * len_scalue_e
            losses.append(loss)
            yprev = ystep
            # ss
            if self.ss_rate < 1.0:
                remain_probs = utils.Random.binomial(1, self.ss_rate, bsize, "ss")
                remain_flags = [x>=0.5 for x in remain_probs]
                # todo(warn): accept err tokens
                results_topk = layers.BK.topk(scores_exprs, self.ss_size)
                for ii in range(bsize):
                    if not remain_flags[ii]:
                        rr = results_topk[ii]
                        values = np.asarray([x[VAL_DIM] for x in rr])
                        probs = np.exp(values) / np.sum(np.exp(values), axis=0)
                        selected = utils.Random.multinomial_select(probs, "ss")
                        next_y = rr[selected][IDX_DIM]
                        yprev[ii] = next_y
            # prepare next steps: only following gold
            new_opens = [State(prev=ss, action=Action(yy, 0.)) for ss, yy in zip(opens, yprev)]
            opens = new_opens
        # -- final
        loss0 = layers.BK.esum(losses)
        loss = layers.BK.sum_batches(loss0) / bsize
        if training:
            layers.BK.forward(loss)
            layers.BK.backward(loss)
        # return value?
        if ret_value == "loss":
            lossy_val = layers.BK.get_value_sca(loss)
            return {"y": lossy_val*bsize}
        elif ret_value == "losses":
            # return token-wise loss
            origin_values = [layers.BK.get_value_vec(i) for i in losses]
            reshaped_values = [[origin_values[j][i] for j in range(yl)] for i, yl in enumerate(ylens)]
            if self.get_prop("r2l"):
                reshaped_values = [list(reversed(one[:-1]))+[one[-1]] for one in reshaped_values]
            return reshaped_values
        else:
            return {}

    # ----- losses -----

    def _mlev_loss_step(self, scores_exprs, ystep, mask_expr):
        # (wasted) getting values to test speed
        # gold_exprs = layers.BK.pick_batch(scores_exprs, ystep)
        # gold_vals = layers.BK.get_value_vec(gold_exprs)
        # pp = layers.BK.topk(scores_exprs, 1)
        pp = layers.BK.topk(scores_exprs, 8)
        # max_exprs = layers.BK.max_dim(scores_exprs)
        # scores_exprs.forward()
        # max_tenidx0 = scores_exprs.tensor_value()
        # zz = max_tenidx0.argmax()
        # max_tenidx = scores_exprs.tensor_value().argmax()
        return self._mle_loss_step(scores_exprs, ystep, mask_expr)

    def _mle_loss_step(self, scores_exprs, ystep, mask_expr):
        if self.margin_ > 0.:
            scores_exprs = layers.BK.add_margin(scores_exprs, ystep, self.margin_)
        one_loss = layers.BK.pickneglogsoftmax_batch(scores_exprs, ystep)
        if mask_expr is not None:
            one_loss = one_loss * mask_expr
        return one_loss

    def _hinge_max_loss_step(self, scores_exprs, ystep, mask_expr):
        if self.margin_ > 0.:
            scores_exprs_final = layers.BK.add_margin(scores_exprs, ystep, self.margin_)
        else:
            scores_exprs_final = scores_exprs
        # max_exprs = layers.BK.max_dim(scores_exprs)
        max_idxs = layers.BK.topk(scores_exprs_final, 1, prepare=False)[IDX_DIM]
        max_exprs = layers.BK.pick_batch(scores_exprs_final, max_idxs)
        gold_exprs = layers.BK.pick_batch(scores_exprs_final, ystep)
        # get loss
        one_loss = max_exprs - gold_exprs
        if mask_expr is not None:
            one_loss = one_loss * mask_expr
        return one_loss

    def _hinge_avg_loss_step(self, scores_exprs, ystep, mask_expr):
        one_loss_all = layers.BK.hinge_batch(scores_exprs, ystep, self.margin_)
        ## todo: approximate counting code here (-3 for squeezing negatives, +1 for smoothing)
        # -- still out of memory on 12g gpu, maybe need smaller bs
        gold_exprs = layers.BK.pick_batch(scores_exprs, ystep)
        gold_exprs -= (self.margin_ - 3)    # to get to zero for sigmoid
        scores_exprs -= gold_exprs
        count_exprs = layers.BK.sum_elems(layers.BK.logistic(scores_exprs))
        count_exprs = layers.BK.nobackprop(count_exprs)
        one_loss = layers.BK.cdiv(one_loss_all, count_exprs+1)
        if mask_expr is not None:
            one_loss = one_loss * mask_expr
        return one_loss

    def _hinge_avg0_loss_step(self, scores_exprs, ystep, mask_expr):
        one_loss_all = layers.BK.hinge_batch(scores_exprs, ystep, self.margin_)
        gold_exprs = layers.BK.pick_batch(scores_exprs, ystep)
        gold_exprs_minus = gold_exprs - self.margin_
        counts = layers.BK.count_larger(scores_exprs, gold_exprs_minus)
        counts_expr = layers.BK.inputVector_batch(counts)
        one_loss = layers.BK.cdiv(one_loss_all, counts_expr)
        if mask_expr is not None:
            one_loss = one_loss * mask_expr
        return one_loss

    def _hinge_sum_loss_step(self, scores_exprs, ystep, mask_expr):
        one_loss = layers.BK.hinge_batch(scores_exprs, ystep, self.margin_)
        if mask_expr is not None:
            one_loss = one_loss * mask_expr
        return one_loss

    # ----- losses -----

    # todo(warn): advanced training process, similar to the mt_search part, but rewrite to avoid messing up, which
    # -> is really not a good idea, and this should be combined with mt_search, however ...
    # -> do not re-use the decoding part of opts, rename those to t2_*

    def fb_branch_(self, insts, training, ret_value):
        # this branches from gold which is different from the search_branch_ which branches from greedy-best
        raise NotImplementedError("To be implemented")
        pass

    def fb_beam_(self, insts, training, ret_value):
        # pieces = self.opts["t2_beam_size"]
        # if self.opts["t2_gold_run"]:
        #     pieces += 1
        # real_bsize = len(insts)
        # impl_bsize = real_bsize // pieces
        real_bsize = len(insts)
        impl_bsize = self.opts["t2_impl_bsize"]
        v = 0.
        idx = 0
        while idx < real_bsize:
            # reuse masks, todo(warn) always new_graph for this one
            Model.new_graph()
            self.refresh(training)
            idx_bound = min(idx+impl_bsize, real_bsize)
            v += self.fb_beam_impl_(insts[idx:idx_bound], real_bsize, training, ret_value)
            idx += impl_bsize
        return {"y":v}

    def fb_beam_impl_(self, insts, real_bsize, training, ret_value):
        # always keeping the same size for more efficient batching
        utils.zcheck(training, "Only for training mode.")
        xs, ys = self.prepare_xy_(insts)
        # --- no need to refresh
        # -- options and vars
        # basic
        bsize = len(xs)
        esize_all = self.opts["t2_beam_size"]
        ylens = [len(_y) for _y in ys]
        ylens_max = [int(np.ceil(_yl*self.opts["t2_search_ratio"])) for _yl in ylens]   # todo: currently, just cut off according to ref length
        # pruners (no diversity penalty here for training)
        need_att = (self.opts["cov_record_mode"] != "none")
        # -> local
        t2_local_expand = max(2, min(self.opts["t2_local_expand"], esize_all))      # todo(warn): could have err-states
        t2_local_diff = self.opts["t2_local_diff"]
        # -> global beam/gold merging for ngram
        t2_global_expand = self.opts["t2_global_expand"]
        t2_global_diff = self.opts["t2_global_diff"]
        t2_bngram_n = self.opts["t2_bngram_n"]
        t2_bngram_range = self.opts["t2_bngram_range"]
        # -> sync
        t2_sync_med = self.opts["t2_sync_med"]
        t2_med_range = self.opts["t2_med_range"]
        t2_sync_nga = self.opts["t2_sync_nga"]
        t2_nga_n = self.opts["t2_nga_n"]
        t2_nga_range = self.opts["t2_nga_range"]
        # update
        t2_beam_up = self.opts["t2_beam_up"]
        #
        EOS_ID = self.target_dict.eos
        model_softmax = not self.get_prop("no_model_softmax")
        t2_gold_run = self.opts["t2_gold_run"]
        t2_gi_mode = self.opts["t2_gi_mode"]
        t2_beam_nodrop = self.opts["t2_beam_nodrop"]
        # => START
        State.reset_id()
        f_new_sg = lambda _i: SearchGraph(target_dict=self.target_dict, src_info=insts[_i])
        # 1. first running the gold seqs if needed
        gold_cur = [State(sg=f_new_sg(_i)) for _i in range(bsize)]
        gold_states = [gold_cur]
        gold_ends = [None] * bsize  # un-sync ends
        running_yprev = None
        running_cprev = None
        for step_gold in range(max(ylens)):
            # running the caches
            ystep, mask_expr = prepare_y_step(ys, step_gold)    # set 0 for padding if ended
            # select the next steps anyway
            gold_next = []
            for _i in range(bsize):
                x = gold_cur[_i]
                if x is PADDING_STATE or x.action_code == EOS_ID:
                    gold_next.append(PADDING_STATE)
                else:
                    next_x = State(prev=x, action=Action(ystep[_i], 0.))
                    gold_next.append(next_x)
                    if next_x.action_code == EOS_ID:
                        gold_ends[_i] = next_x
            if t2_gold_run and not t2_beam_nodrop:  # todo(warn): rerun later with dropouts
                # softmax for the "correct" score if needed
                if step_gold == 0:
                    cc = self.start(xs, softmax=model_softmax)
                else:
                    cc = self.step(running_cprev, running_yprev, gold_cur, softmax=model_softmax)
                running_yprev = ystep
                running_cprev = cc
                # attach caches
                sc_expr = layers.BK.pick_batch(cc["results"], ystep)
                sc_val = layers.BK.get_value_vec(sc_expr)
                if need_att:
                    atts_v = layers.BK.get_value_np(cc["att"])
                # todo(warn): here not calling explain for simplicity
                if model_softmax:
                    sc_val = np.log(sc_val)
                for _i in range(bsize):
                    if gold_cur[_i] is not PADDING_STATE:
                        gold_cur[_i].set(VNAME_CACHE, (cc, _i))
                        # todo(warn): not updating the accumulated scores
                        if gold_next[_i] is not PADDING_STATE:
                            gold_next[_i].action_score(sc_val[_i])
                            if need_att:
                                gold_next[_i].set(VNAME_ATTW, atts_v[_i])
            gold_cur = gold_next
            gold_states.append(gold_next)
        # 2. then start the beam search (with the knowledge of gold-seq)
        if t2_beam_nodrop:
            Model.new_graph()
            self.refresh(False)     # search with no dropouts
        beam_states = []
        beam_cur = [[PADDING_STATE for _j in range(esize_all)] for _i in range(bsize)]
        beam_states.append(beam_cur)
        beam_ends = [[] for _i in range(bsize)]
        beam_remains = [esize_all for _i in range(bsize)]
        beam_update_points = [[] for _i in range(bsize)]        # todo(warn): currently only recording the best points
        for _i in range(bsize):     # set the init states
            one = State(sg=f_new_sg(_i))
            one.set(VNAME_GOLD, True)                    # init as zero gold
            one.set(VNAME_ATT, gold_states[0][_i])       # attach to init gold
            one.set(VNAME_ATT_BASE, 0)                   # last attaching length
            one.set(VNAME_SYNCLEN, 0)                    # init attach-based len is 0
            one.set(VNAME_GISTAT, "START")               # gold-inference state: marking updating boundaries
            beam_cur[_i][0] = one
        # ready go, break at the end
        running_yprev = None
        running_cprev = None
        step_beam = 0
        beam_continue_flag = True
        while beam_continue_flag:
            next_beam_cur = [None] * bsize
            beam_continue_flag = False
            # running the caches
            if step_beam == 0:
                if t2_gold_run and not t2_beam_nodrop:
                    # todo(warn): expand previously calculated values and also results
                    cc = self.repeat(gold_states[0][0].get(VNAME_CACHE)[0], bsize, esize_all, {"hid", "S", "V", "out_s", "results"})
                else:
                    cc = self.start(xs, repeat_time=esize_all, softmax=model_softmax)
            else:
                cc = self.step(running_cprev, running_yprev, beam_cur, softmax=model_softmax)
            # attach caches
            for _i in range(bsize):
                for _j in range(esize_all):
                    one = beam_cur[_i][_j]
                    if one is not PADDING_STATE:
                        one.set(VNAME_CACHE, (cc, _i*esize_all+_j))
            # compare for the next steps --- almost same as beam_search, but all-batched and simplified
            results_topk = layers.BK.topk(cc["results"], t2_local_expand)
            if need_att:
                atts_v = layers.BK.get_value_np(cc["att"])
            for i in range(bsize):
                # collect local candidates
                global_cands = []
                for j in range(esize_all):
                    prev = beam_cur[i][j]
                    # skip ended states
                    if prev is PADDING_STATE or prev.is_end():
                        continue
                    inbatch_idx = i*esize_all+j
                    rr0 = results_topk[inbatch_idx]
                    rr = self.explain_result_topkp(rr0)
                    local_cands = []
                    for onep in rr:
                        if need_att:
                            local_one = State(prev=prev, action=Action(onep[IDX_DIM], onep[VAL_DIM]), attw=atts_v[inbatch_idx])
                        else:
                            local_one = State(prev=prev, action=Action(onep[IDX_DIM], onep[VAL_DIM]))
                        local_cands.append(local_one)
                    survived_local_cands = SPruner.local_prune(local_cands, t2_local_expand, t2_local_diff, 0.)
                    global_cands += survived_local_cands
                # sort them all
                global_cands.sort(key=(lambda x: x.score_partial), reverse=True)
                # global pruning (if no bngram, then simply get the first remains[i] ones)
                # TODO (using some default values for _score_f here)
                survived_global_cands = SPruner.global_prune_ngram_greedy(cand_states=global_cands, rest_beam_size=beam_remains[i], sig_beam_size=t2_global_expand, thresh=t2_global_diff, penalty=0., ngram_n=t2_bngram_n, ngram_range=t2_bngram_range)
                # =========================
                # setting states properties
                gold_index = -1
                gold_prev_index = -1
                cur_ys = ys[i]
                for j, one in enumerate(beam_cur[i]):
                    if one is not PADDING_STATE:
                        if one.get(VNAME_GOLD):
                            gold_prev_index = j
                for j, one in enumerate(survived_global_cands):
                    # GOLD_NAME
                    if one.prev.get(VNAME_GOLD):
                        if one.length<=len(cur_ys) and one.action_code==cur_ys[one.length-1]:
                            one.set(VNAME_GOLD, True)
                            gold_index = j
                    # ATTACH_NAME
                    default_len = one.prev.get(VNAME_SYNCLEN)+1
                    default_attbase = one.prev.get(VNAME_ATT_BASE)
                    att_len_min = default_attbase+1
                    if t2_sync_nga:
                        # only attach ahead, no way to look backward
                        for g_step in reversed(range(max(att_len_min, default_len-t2_nga_range+1), min(len(cur_ys), default_len+t2_nga_range))):
                            g_one = gold_states[g_step][i]
                            if g_one.sig_ngram(t2_nga_n) == one.sig_ngram(t2_nga_n):
                                # hit for ngram merging, mark it & break
                                one.set(VNAME_ATT, g_one)
                                default_attbase = g_step
                                default_len = g_step
                    one.set(VNAME_SYNCLEN, default_len)
                    one.set(VNAME_ATT_BASE, default_attbase)
                # =========================
                # gold interfering (also recording the interfering point for updating)
                # gi1: ngram gold attach: find ngram to attach on the gold seq
                if len(survived_global_cands) > 0:
                    if t2_gi_mode == "ngab":
                        # only remain the gold-merged 1st-ranked one
                        if survived_global_cands[0].get(VNAME_ATT) is not None:
                            beam_update_points[i].append(survived_global_cands[0])
                            survived_global_cands = [survived_global_cands[0]]
                            survived_global_cands[0].set(VNAME_GISTAT, "NGAB")
                    # gi2: laso: re-pick the pruned gold state
                    elif t2_gi_mode == "laso":
                        # re-pick the gold seq
                        if gold_index < 0:
                            beam_update_points[i].append(survived_global_cands[0])
                            utils.zcheck(gold_prev_index>=0, "Unreasonable non-existing prev-gold for the laso mode!!")
                            prev_gold_ss = beam_cur[i][gold_prev_index]
                            according_gold = gold_states[prev_gold_ss.length+1][i]
                            repicked_gold = State(prev=prev_gold_ss, action=according_gold.action)
                            repicked_gold.set(VNAME_GOLD, True)
                            repicked_gold.set(VNAME_ATT, according_gold)
                            repicked_gold.set(VNAME_ATT_BASE, repicked_gold.length)
                            repicked_gold.set(VNAME_SYNCLEN, repicked_gold.length)
                            repicked_gold.set(VNAME_GISTAT, "LASO")
                            if survived_global_cands[0].get(VNAME_ATT) is None:
                                survived_global_cands[0].set(VNAME_ATT, according_gold)
                            survived_global_cands = [repicked_gold]
                    # nope: nothing to do
                    else:
                        pass
                # =========================
                # force to eos if out of maxlen & add eos ones to ends
                force_eos_thresh = ylens_max[i]
                cur_gold_end = gold_ends[i]
                for one in survived_global_cands:
                    # eos forcing
                    force_ending = False
                    if one.action_code != EOS_ID and one.length >= force_eos_thresh:
                        # one.action_score(0.)
                        # one.action = Action(EOS_ID, 0.)
                        force_ending = True
                    # eos recording
                    if (one.action_code == EOS_ID or force_ending) and beam_remains[i]>0:
                        one.mark_end()
                        one.set(VNAME_ATT, cur_gold_end)
                        one.set(VNAME_ATT_BASE, cur_gold_end.length)
                        one.set(VNAME_SYNCLEN, cur_gold_end.length)
                        # (nope) one.set(VNAME_GISTAT, "END")
                        beam_remains[i] -= 1
                        beam_ends[i].append(one)
                #
                if len(survived_global_cands) > 0:
                    beam_continue_flag = True
                while len(survived_global_cands) < esize_all:
                    survived_global_cands.append(PADDING_STATE)
                next_beam_cur[i] = survived_global_cands
            # ========================= outside recording and preparing
            running_yprev = []
            running_cprev_orders = []
            for _i in range(bsize):
                once_cands = next_beam_cur[_i]
                for _j in range(esize_all):
                    one_cand = once_cands[_j]
                    if one_cand is None:
                        running_yprev.append(PADDING_ACTION)
                        running_cprev_orders.append(_i*esize_all)
                    else:
                        running_yprev.append(one_cand.action_code)
                        running_cprev_orders.append(one_cand.prev.get(VNAME_CACHE)[-1])
            # todo(warn): always pad the beam, thus None for bi_orders
            if esize_all > 1:
                running_cprev = self.rerange(cc, running_cprev_orders, None)
            else:
                running_cprev = cc
            # how to break -> flag
            step_beam += 1
            beam_states.append(next_beam_cur)
            beam_cur = next_beam_cur
        # ===============================
        # -- ranking the best searched ones
        if self.opts["t2_compare_at"] == "norm":
            _get_score_f = (lambda x: x.score_partial/x.length)
        else:
            _get_score_f = (lambda x: x.score_partial)
        for _i in range(bsize):
            best_end = sorted(beam_ends[_i], key=_get_score_f, reverse=True)[0]
            if len(beam_update_points[_i])<=0 or best_end.length > beam_update_points[_i][-1].length:
                beam_update_points[_i].append(best_end)
        # ================================
        # running gold states or re-run beam states with dropouts
        if t2_beam_nodrop:
            Model.new_graph()
            self.refresh(training)     # run with dropouts when trained
            # re-run and re-attach caches with dropouts
            rerun_states = []
            rerun_ys = []
            for _i in range(bsize):
                utils.zcheck(len(beam_update_points[_i])==1, "Currently unsupport rerun with more than one update points!!")
                this_ends = [beam_update_points[_i][0]]
                if t2_gold_run:
                    this_ends.append(gold_ends[_i])
                for one_end in this_ends:
                    one_path = one_end.get_path()
                    one_ys = [x.action_code for x in one_path]
                    one_path = [one_path[0].prev] + one_path
                    rerun_states.append(one_path)
                    rerun_ys.append(one_ys)
            # rerun
            rerun_bsize = len(rerun_ys)
            utils.zcheck(rerun_bsize%bsize==0, "Strange rerun bsize")
            expand_size = rerun_bsize // bsize
            #
            rerun_ylens = [len(_y) for _y in rerun_ys]
            rerun_maxlen = max(rerun_ylens) + 1     # todo(warn) +1 to get caches for the last state, which might be rebased
            rerun_prevc, rerun_prevy = None, None
            for rerun_i in range(rerun_maxlen):
                ystep, mask_expr = prepare_y_step(rerun_ys, rerun_i)
                if rerun_i == 0:
                    cc = self.start(xs, repeat_time=expand_size, softmax=False)
                else:
                    cc = self.step(rerun_prevc, rerun_prevy, None, softmax=False)
                rerun_prevc = cc
                rerun_prevy = ystep
                # todo(warn): about the scores and attw
                # assign caches
                for tmp_idx in range(rerun_bsize):
                    if rerun_i < len(rerun_states[tmp_idx]):
                        rerun_states[tmp_idx][rerun_i].set(VNAME_CACHE, (cc, tmp_idx))
        # ================================
        # -- collecting loss according to update_points and use masks for batching
        self.beam_losser_.refresh()
        for _i in range(bsize):
            # preparing update points
            update_points = beam_update_points[_i]
            if t2_beam_up == "fg":
                # only updating at the first one, wasting much time for this mode
                update_points = update_points[0]
            # collect them
            for up in reversed(update_points):
                self.beam_losser_.build_one(up, 1.0)    # todo: scale for the loss
        losses = self.beam_losser_.get_loss(self)    # one value for sum of the batches
        loss = losses / real_bsize               # real batch size
        if training:
            layers.BK.forward(loss)
            layers.BK.backward(loss)
        # return value?
        if ret_value == "loss":
            lossy_val = layers.BK.get_value_sca(loss)
            return lossy_val*real_bsize
        else:
            return 0.

# -- build loss for fb_beam (collecting loss according to predicted ones)
class BeamLossBuilder(object):
    def __init__(self, opts, vocab):
        self.opts = opts
        self.vocab = vocab
        self.ERR_ID = vocab.err
        self.IGNORE_IDS = vocab.get_ending_tokens()
        #
        self.t2_sync_nga = opts["t2_sync_nga"]
        self.t2_sync_med = opts["t2_sync_med"]
        if self.t2_sync_med:
            self.seq_build_f = self.build_seq_med
        elif self.t2_sync_nga:
            self.seq_build_f = self.build_seq_nga
        else:
            self.seq_build_f = self.build_seq_default
        #
        self.t2_beam_loss = opts["t2_beam_loss"]
        self.t2_bad_lambda = opts["t2_bad_lambda"]
        self.t2_bad_maxlen = opts["t2_bad_maxlen"]
        if self.t2_beam_loss == "per":
            self.loss_build_f = self.build_loss_per
            self.per_norm = opts["t2_compare_at"]=="norm"
            self.margin = opts["train_margin"]
        elif self.t2_beam_loss == "err":
            self.loss_build_f = self.build_loss_err
            self.t2_err_gold_lambda = opts["t2_err_gold_lambda"]
            self.t2_err_pred_lambda = opts["t2_err_pred_lambda"]
            # error state's corresponding gold seq's mode
            self.t2_err_gold_mode = opts["t2_err_gold_mode"]
            self.eg_mode_gold = (self.t2_err_gold_mode=="gold")
            self.eg_mode_based = (self.t2_err_gold_mode=="based")
            # for the matched or correction part
            self.t2_err_match_nope = opts["t2_err_match_nope"]
            self.t2_err_match_addfirst = opts["t2_err_match_addfirst"]
            self.t2_err_match_addeos = opts["t2_err_match_addeos"]
            self.t2_err_cor_nofirst = opts["t2_err_cor_nofirst"]
            # matched thresholds
            self.t2_err_seg_minlen = opts["t2_err_seg_minlen"]
            self.t2_err_mcov_thresh = opts["t2_err_mcov_thresh"]
            self.t2_err_pright_thresh = opts["t2_err_pright_thresh"]
            self.t2_err_thresh_bleu = opts["t2_err_thresh_bleu"]
            #
            self.t2_err_debug_print = opts["t2_err_debug_print"]
        else:
            self.loss_build_f = None
        self.med_ifsub = not opts["t2_med_nosub"]
        self.med_seg_merger = MedSegMerger(opts, vocab)
        #
        self.refresh(True)

    # --------
    # refresh
    def refresh(self, clear_stat=False):
        # clear and start a new loss-building process
        self.loss_maps = {}
        self.rebased_maps = []      # special ones for running re-based ones (rebased-basic-cache, tokens)
        if clear_stat:
            # stats
            self.num_points = 0
            self.num_points_valid = 0   # points that are matched above thresh-2
            self.num_points_pright = 0  # points turned to gold because matched too much
            self.num_segs = 0
            self.num_tokens = 0
            self.num_bad_tokens = 0
            self.num_bad_tokens_rebasable = 0   # bad tokens following non-gold seq (rebase and gold will be different)
            self.num_good_tokens = 0
            self.num_good_tokens_nostarts = 0   # good tokens not at start
            # stats2
            # err
            self.num_tok_mod = 0        # modified (first err)
            self.num_tok_err = 0        # err states (rest err)
            self.num_tok_good = 0       # matched
            self.num_tok_gold = 0       # gold running
            self.num_tok_rebased = 0    # rebased gold running
            self.num_tok_falsematch = defaultdict(int)     # regard as unmatched under thresh-1 (counted as bad_tokens)
            # err stat
            self.match_cov_stater = utils.RangeStater(0., 1., 10)
            self.match_seg_stater = utils.RangeStater(1, 20, 10)
            # per
            self.num_tok_minus = 0
            self.num_tok_plus = 0

    def report(self):
        # r = {}
        # for n in self.__dict__:
        #     if n.startswith("num"):
        #         r[n] = self.__dict__[n]
        # return r
        s = ""
        divisor = self.num_points_valid + 1e-5
        divisor0 = self.num_points + 1e-5
        s += "Insts:%d/%d/%d=%.3f/%.3f||Segs:%d(%.3f)||Toks:%d(%.3f)->match:%d(%.3f)/ns:%d(%.3f),non:%d(%.3f)/rebasable:%d(%.3f)||" \
             % (self.num_points_pright, self.num_points_valid, self.num_points, self.num_points_pright/divisor0, self.num_points_valid/divisor0,
                self.num_segs, self.num_segs/divisor, self.num_tokens, self.num_tokens/divisor,
                self.num_good_tokens, self.num_good_tokens/divisor, self.num_good_tokens_nostarts, self.num_good_tokens_nostarts/divisor,
                self.num_bad_tokens, self.num_bad_tokens/divisor, self.num_bad_tokens_rebasable, self.num_bad_tokens_rebasable/divisor,)
        if self.t2_beam_loss == "per":
            s += "Perc->minus(beam):%d/%.3f,plus(gold):%d/%.3f" \
                 % (self.num_tok_minus, self.num_tok_minus/divisor, self.num_tok_plus, self.num_tok_plus/divisor)
        elif self.t2_beam_loss == "err":
            falsematch_s0 = "/".join("%s:%d"%(k,v) for k,v in self.num_tok_falsematch.items())
            falsematch_s1 = "/".join("%s:%.3f"%(k,v/divisor) for k,v in self.num_tok_falsematch.items())
            s += "Err->err0:%d/%.3f,err:%d/%.3f,match:%d/%.3f,gold:%d/%.3f,rebased:%d/%.3f,falsematch:%s/%s" \
                 % (self.num_tok_mod, self.num_tok_mod/divisor, self.num_tok_err, self.num_tok_err/divisor,
                    self.num_tok_good, self.num_tok_good/divisor, self.num_tok_gold, self.num_tok_gold/divisor,
                    self.num_tok_rebased, self.num_tok_rebased/divisor, falsematch_s0, falsematch_s1)
            s += "||MC-ranges:%s, Seg-ranges:%s" % (self.match_cov_stater.descr(), self.match_seg_stater.descr())
        else:
            pass
        return s
    # --------

    # BeamLossBuilder.pretty_descr(segs,True)
    @staticmethod
    def pretty_descr(segs, print=False):
        s = ""
        if len(segs[0][1])>0:
            s += str(segs[0][1][0].sg.src_info) + "\n"
        else:
            s += str(segs[0][2][0].sg.src_info) + "\n"
        for one in reversed(segs):
            if one[0]:
                s += "YEP: \n"
            else:
                s += "NOP: \n"
            for rr in [reversed(one[1]), reversed(one[2])]:
                for ss in rr:
                    s += ss.descr_word() + " "
                s += "\n"
        if print:
            utils.zlog(s)
        return s

    # build_one
    def build_one(self, point, scale):
        self.num_points += 1
        ori_segs = self.seq_build_f(point)
        for ss in ori_segs:
            if ss[0]:   # stat
                self.match_seg_stater.add(len(ss[1]))
        # ** Thresh 1: length of the matched segment
        # merge short & maybe unfit segments
        segs, cur_falsematch = self.med_seg_merger.merge(ori_segs)
        # count matches
        if self.t2_err_thresh_bleu:
            tokens_beam = [one.action_code for one in point.get_path()]
            tokens_gold = [one.action_code for one in point.get("AT").get_path()]
            matched_cov = bleu_single(tokens_beam, [tokens_gold])
        else:
            len_all, len_match = 0, 0
            for ss in segs:
                len_all += len(ss[1])
                if ss[0]:
                    len_match += len(ss[1])
            matched_cov = len_match/len_all
        self.match_cov_stater.add(matched_cov)
        # debug print
        if self.t2_err_debug_print:
            utils.zlog("===== Another match-cov is zz%.3f" % matched_cov)
            BeamLossBuilder.pretty_descr(segs, True)
        # ** Thresh 2: matched ratio
        # todo(warn): currently(18.01.18) no longer consider multiple update points
        if matched_cov >= self.t2_err_pright_thresh:
            self.num_points_pright += 1
            self.build_loss_ordinary(point, scale)
        elif matched_cov >= self.t2_err_mcov_thresh:
            length_segs = len(segs)
            prev_seg = None
            self.num_points_valid += 1
            for k in cur_falsematch:
                self.num_tok_falsematch[k] += cur_falsematch[k]
            bad_flag = False
            for idx in range(length_segs):
                real_idx = length_segs-1-idx
                this_seg, next_seg = segs[real_idx], segs[real_idx-1]
                if real_idx <= 0:
                    next_seg = None
                isMatch, beamSeg, goldSeg = segs[length_segs-1-idx]
                # stat
                self.num_segs += 1
                self.num_tokens += len(beamSeg)
                if isMatch:
                    self.num_good_tokens += len(beamSeg)
                    if idx != 0:
                        self.num_good_tokens_nostarts += len(beamSeg)
                else:
                    self.num_bad_tokens += len(beamSeg)
                    if bad_flag:
                        self.num_bad_tokens_rebasable += len(beamSeg)
                    bad_flag = True
                # record into maps
                self.loss_build_f(this_seg, prev_seg, next_seg, scale)
                #
                prev_seg = this_seg
        else:
            # give up the matched segments
            whole_seg = self.build_seq_default(point)
            self.loss_build_f(whole_seg[0], None, None, scale)

    # loss
    def get_loss(self, mm):
        def _sum_all(es):
            if len(es) > 0:
                return layers.BK.sum_batches(layers.BK.esum(es))
            else:
                utils.zcheck(False, "Empty list, non loss for this list.", func="warn")
                return layers.BK.zeros((1,))
        # first gather as batched
        loss_idxs = BeamLossBuilder.gather_loss_from_map(self.loss_maps)
        if self.t2_beam_loss == "per":
            losses = [layers.BK.pick_batch(e,i)*layers.BK.inputVector_batch(m) for e,i,m in loss_idxs]
            return _sum_all(losses)
        elif self.t2_beam_loss == "err":
            v = layers.BK.zeros((1,))
            if len(loss_idxs) > 0:
                losses = [layers.BK.pickneglogsoftmax_batch(e,i)*layers.BK.inputVector_batch(m) for e,i,m in loss_idxs]
                v += _sum_all(losses)
            if self.eg_mode_based or len(self.rebased_maps)>0:  # need to calc
                rebased_losses = self.get_rebased_gold_losses(mm)
                v += self.t2_err_gold_lambda * _sum_all(rebased_losses)
            return v
        else:
            raise NotImplementedError("Unknown beam loss %s." % (self.t2_beam_loss,))

    # build rebased loss -> like running a gold seq
    def get_rebased_gold_losses(self, mm):
        self.rebased_maps.sort(key=lambda x: len(x[-1]))    # sorting them for better batching
        batching_width = layers.BK.bsize(self.rebased_maps[0][0].get(VNAME_CACHE)[0]["out_s"])
        cur_idx = 0
        bound_idx = len(self.rebased_maps)
        losses = []
        while cur_idx < bound_idx:
            cur_rebases = self.rebased_maps[cur_idx:min(cur_idx+batching_width, bound_idx)]
            start_caches, start_cidxes = [x[0].get(VNAME_CACHE)[0] for x in cur_rebases], [x[0].get(VNAME_CACHE)[1] for x in cur_rebases]
            ys = [x[1] for x in cur_rebases]
            # -> almost same as fb_std
            ylens = [len(_y) for _y in ys]
            cur_maxlen = max(ylens)
            caches = []
            yprev = None
            for i in range(cur_maxlen):
                ystep, mask_expr = prepare_y_step(ys, i)
                if i==0:
                    cc = mm.recombine(start_caches, start_cidxes, ["hid", "S", "V", "out_s"])
                else:
                    cc = mm.step(caches[-1], yprev, None, softmax=False)
                caches.append(cc)
                scores_exprs = cc["out_s"]
                loss = mm.losser_(scores_exprs, ystep, mask_expr)
                losses.append(loss)
                yprev = ystep
            cur_idx += batching_width
        return losses

    # ---------- specific ones ---------- #
    # update_points -> attaching sequences (reversed one)
    def build_seq_default(self, point):
        cur, cur_at = point, point.get(VNAME_ATT)
        list_beam, list_att = [cur], [cur_at]
        cur = cur.prev
        cur_at = cur_at.prev
        #
        while cur.get(VNAME_GISTAT) is None:
            list_beam.append(cur)
            cur = cur.prev
        #
        att_end = cur.get(VNAME_ATT)
        while cur_at.length > att_end.length:
            list_att.append(cur_at)
            cur_at = cur_at.prev
        return [(False, list_beam, list_att)]

    def build_seq_nga(self, point):
        def _add_if_nonempty(sgs, one):
            if len(one[1])>0 or len(one[2])>0:
                sgs.append(one)
        def _add_until(cur_at, trg_at, one):
            while cur_at.length > trg_at.length:
                one[2].append(cur_at)
                cur_at = cur_at.prev
            return cur_at
        segs = []    # list of (ifMatch, [beam-states], [gold-states])
        # get the interval
        list_beam = [point]
        list_att = [point.get(VNAME_ATT)]
        list_att_notnone = [point.get(VNAME_ATT)]
        cur = point.prev
        while cur.get(VNAME_GISTAT) is None:
            list_beam.append(cur)
            cur_at = cur.get(VNAME_ATT)
            list_att.append(cur_at)
            if cur_at is not None:
                list_att_notnone.append(cur_at)
            cur = cur.prev
        list_att_notnone.reverse()  # list_att_notnone[-1] is the latest one
        att_bound = cur.get(VNAME_ATT)
        att_prev_length = 0
        # scan again
        cur_at = list_att[0]    # must be there
        idx = 0
        while idx < len(list_beam):
            # eat matched seg
            s0 = (True, [], [])
            while idx < len(list_beam):
                p = list_beam[idx]
                at = list_att[idx]
                # new attachment
                if at is not None:
                    # maybe skip one unmatched gold-only-seq
                    if at.length < cur_at.length:
                        _add_if_nonempty(segs, s0)
                        s0 = (True, [], [])
                        s01 = (False, [], [])
                        cur_at = _add_until(cur_at, at, s01)
                        _add_if_nonempty(segs, s01)
                    cur_at = at
                    # previous at's constrain (there can be repeated two-way matches, prefer the previous one)
                    while len(list_att_notnone) > 0 and list_att_notnone[-1].length >= at.length:
                        list_att_notnone = list_att_notnone[:-1]
                    if len(list_att_notnone) > 0:
                        att_prev_length = list_att_notnone[-1].length
                    else:
                        att_prev_length = 0
                # check equal
                if cur_at.action_code == p.action_code and cur_at.length > max(att_prev_length, att_bound.length):
                    s0[1].append(p)
                    s0[2].append(cur_at)
                    idx += 1
                    cur_at = cur_at.prev
                else:
                    break
            _add_if_nonempty(segs, s0)
            # eat unmatched seg
            s1 = (False, [], [])
            while idx < len(list_beam):
                p = list_beam[idx]
                at = list_att[idx]
                # break to next att
                if at is None or at.action_code != p.action_code:
                    s1[1].append(p)
                    idx += 1
                else:
                    cur_at = _add_until(cur_at, at, s1)
                    break
            if idx >= len(list_beam):
                # fill in extra gold seqs at the final step
                cur_at = _add_until(cur_at, att_bound, s1)
            _add_if_nonempty(segs, s1)
        return segs

    def build_seq_med(self, point):
        # obtain the two list
        list_beam = [point]
        list_gold = [point.get(VNAME_ATT)]
        cur = point.prev
        while cur.get(VNAME_GISTAT) is None:
            list_beam.append(cur)
            cur = cur.prev
        end_att = cur.get(VNAME_ATT)
        cur_att = list_gold[0].prev
        while cur_att.length > end_att.length:
            list_gold.append(cur_att)
            cur_att = cur_att.prev
        tokens_beam = [one.action_code for one in reversed(list_beam)]
        tokens_gold = [one.action_code for one in reversed(list_gold)]
        # dp for med
        segs = do_med(tokens_beam, tokens_gold, list_beam, list_gold, self.med_ifsub)
        return segs

    # --------------
    @staticmethod
    def add_loss_to_map(m, cache, idx, token, scale):
        # id -> (cache, {idx -> (token, scale)})
        utils.zcheck(scale>0, "Bad scale %s" % scale, func="warn")
        cid = id(cache)
        if cid in m:
            m2 = m[cid][1]
            if idx in m2:
                v = m2[idx]
                if v[0] != token:
                    utils.zcheck(False, "Ignoring: unsupported multiple token loss: %s vs %s" % (v, [token, scale]), func="warn")
                else:
                    utils.zcheck(False, "Ignoring: strange multiple token loss: %s vs %s" % (v, [token, scale]), func="warn")
            else:
                m2[idx] = [token, scale]
        else:
            m[cid] = (cache, {idx: [token, scale]})

    @staticmethod
    def gather_loss_from_map(m):
        # list of (expr, idx, scales)
        ret = []
        for cid in m:
            cache, idxs = m[cid]
            out_expr = cache["out_s"]
            bsize = layers.BK.bsize(out_expr)
            #
            out_idxs, out_scales = [0]*bsize, [0.]*bsize
            for bid in idxs:
                out_idxs[bid], out_scales[bid] = idxs[bid]
            ret.append((out_expr, out_idxs, out_scales))
        return ret

    # attaching sequences -> loss building
    def build_loss_per(self, this_seg, prev_seg, next_seg, scale):
        # todo: not exactly max-margin
        isMatch, beamSeg, goldSeg = this_seg
        if isMatch:
            # no updates for matched segs
            pass
        else:
            # skip for one-side instances
            if len(goldSeg)<=0 or len(beamSeg)<=0:
                return
            # check score
            score_gold = sum(one.action_score() for one in goldSeg)
            score_beam = sum(one.action_score() for one in beamSeg)
            scale_gold = scale_beam = scale
            if self.per_norm:
                score_gold /= len(goldSeg)
                scale_gold /= len(goldSeg)
                score_beam /= len(beamSeg)
                scale_beam /= len(beamSeg)
            #
            if score_gold - score_beam <= self.margin:
                # add perceptron style loss
                self.num_tok_minus += len(beamSeg)
                self.num_tok_plus += len(goldSeg)
                #
                cur_lambda = 1. * scale_beam
                for _i, one in enumerate(reversed(beamSeg)):
                    # todo(warn): ignore some tokens
                    if one.action_code in self.IGNORE_IDS:
                        continue
                    BeamLossBuilder.add_loss_to_map(self.loss_maps, one.prev.get(VNAME_CACHE)[0], one.prev.get(VNAME_CACHE)[1],
                                                    one.action_code, cur_lambda)
                    cur_lambda *= self.t2_bad_lambda
                    if cur_lambda <= 0. or _i >= self.t2_bad_maxlen:
                        break
                cur_lambda = 1. * scale_gold
                for _i, one in enumerate(reversed(goldSeg)):
                    # todo(warn): ignore some tokens
                    if one.action_code in self.IGNORE_IDS:
                        continue
                    BeamLossBuilder.add_loss_to_map(self.loss_maps, one.prev.get(VNAME_CACHE)[0], one.prev.get(VNAME_CACHE)[1],
                                                    one.action_code, -1*cur_lambda)
                    cur_lambda *= self.t2_bad_lambda
                    if cur_lambda <= 0. or _i >= self.t2_bad_maxlen:
                        break

    def build_loss_err(self, this_seg, prev_seg, next_seg, scale):
        # build mle loss for prev state cache
        isMatch, beamSeg, goldSeg = this_seg
        if isMatch:
            # good states
            if not self.t2_err_match_nope:
                if self.t2_err_match_addfirst:
                    ones = reversed(beamSeg)
                else:   # ignore first tokens
                    ones = reversed(beamSeg[:-1])
                for one in ones:
                    if not self.t2_err_match_addeos and one.action_code in self.IGNORE_IDS:
                        continue    # ignore eos tokens
                    self.num_tok_good += 1
                    BeamLossBuilder.add_loss_to_map(self.loss_maps, one.prev.get(VNAME_CACHE)[0], one.prev.get(VNAME_CACHE)[1],
                                                        one.action_code, scale*self.t2_err_pred_lambda)
        else:
            # correct states or err states
            if len(goldSeg) == 0:
                # find correct token in the next one
                ynext = next_seg[2][-1].action_code
            else:
                ynext = goldSeg[-1].action_code
            # modified one
            cur_lambda = 1. * scale
            cur_idx = len(beamSeg)-1
            cur_thresh = max(0, len(beamSeg) - self.t2_bad_maxlen)
            if cur_idx >= cur_thresh:
                one = beamSeg[cur_idx]
                if not self.t2_err_cor_nofirst:
                    self.num_tok_mod += 1
                    BeamLossBuilder.add_loss_to_map(self.loss_maps, one.prev.get(VNAME_CACHE)[0], one.prev.get(VNAME_CACHE)[1],
                                                    ynext, cur_lambda*self.t2_err_pred_lambda)
                cur_lambda *= self.t2_bad_lambda
                cur_idx -= 1
            # error states
            while cur_lambda > 0. and cur_idx >= cur_thresh:
                one = beamSeg[cur_idx]
                BeamLossBuilder.add_loss_to_map(self.loss_maps, one.prev.get(VNAME_CACHE)[0], one.prev.get(VNAME_CACHE)[1],
                                                    self.ERR_ID, cur_lambda*self.t2_err_pred_lambda)
                cur_lambda *= self.t2_bad_lambda
                self.num_tok_err += 1
                cur_idx -= 1
            # gold states
            if self.eg_mode_gold:
                for one in reversed(goldSeg):
                    self.num_tok_gold += 1
                    BeamLossBuilder.add_loss_to_map(self.loss_maps, one.prev.get(VNAME_CACHE)[0], one.prev.get(VNAME_CACHE)[1],
                                                    one.action_code, scale*self.t2_err_gold_lambda)
            if self.eg_mode_based:
                self.num_tok_rebased += len(goldSeg)
                if len(goldSeg) > 0:
                    if len(beamSeg) == 0:
                        if prev_seg is None:
                            prev_state = next_seg[1][-1].prev
                        else:
                            prev_state = prev_seg[1][0]
                    else:
                        prev_state = beamSeg[-1].prev
                    self.rebased_maps.append((prev_state, [one.action_code for one in reversed(goldSeg)]))

    def build_loss_ordinary(self, gold_end_state, scale):
        for one in gold_end_state.get_path():
            BeamLossBuilder.add_loss_to_map(self.loss_maps, one.prev.get(VNAME_CACHE)[0], one.prev.get(VNAME_CACHE)[1],
                                                    one.action_code, scale*self.t2_err_gold_lambda)
