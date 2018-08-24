# layers for nn

import numpy as np
import _dynet as dy
# todo(warn): init at zl.backends
from zl.backends import BK

# ================= Helpers ====================== #
# get mask inputs
def gen_masks_input(rate, size, bsize):
    return dy.random_bernoulli((size,), 1.-rate, 1./(1.-rate), batch_size=bsize)

def bs(x):
    return x.dim()[1]

# ================= Basic Blocks ================= #
# basic unit (stateful about dropouts)
class Basic(object):
    def __init__(self, model):
        self.model = model
        self.params = {}
        self.iparams = {}
        self.update = None
        # dropouts
        self.hdrop = 0.
        self.idrop = 0.
        self.gdrop = 0.
        self.masks = None
        #
        self._add_params = self._add_parameters

    def _ingraph(self, argv):
        # update means whether the parameters should be updated
        update = bool(argv["update"]) if "update" in argv else True
        ingraph = bool(argv["ingraph"]) if "ingraph" in argv else True
        if ingraph:
            for k in self.params:
                if k.startswith("_"):   # todo(warn) special ones
                    continue
                # todo(warn): dynet changes API
                if not update:
                    raise NotImplementedError("No longer specified here.")
                self.iparams[k] = self.params[k]
                # try:
                #     self.iparams[k] = dy.parameter(self.params[k], update)
                # except NotImplementedError:
                #     if update:
                #         self.iparams[k] = dy.parameter(self.params[k])
                #     else:
                #         self.iparams[k] = dy.const_parameter(self.params[k])
            self.update = update
        # dropouts
        self.hdrop = float(argv["hdrop"]) if "hdrop" in argv else 0.
        self.idrop = float(argv["idrop"]) if "idrop" in argv else 0.
        self.gdrop = float(argv["gdrop"]) if "gdrop" in argv else 0.
        self.masks = None

    def refresh(self, argv):
        raise NotImplementedError("No calling refresh from Basic.")

    # todo(warn), here two kinds of init
    def _add_parameters(self, shape, lookup=False, init="default"):
        if BK.COMMON_CONFIG.enabled:
            return self._add_parameters1(shape, lookup, init)
        else:
            return self._add_parameters0(shape, lookup, init)

    def _add_parameters1(self, shape, lookup=False, init="default"):
        return BK.get_params(self.model, shape, lookup, init)

    # todo(warn): use the original ztry0 version of init
    def _add_parameters0(self, shape, lookup=False, init="default"):
        def ortho_weight(ndim):
            W = np.random.randn(ndim, ndim)
            u, s, v = np.linalg.svd(W)
            return u.astype(np.float)
        def get_init(shape, init):
            # shape is a tuple of dims
            assert init in ["default", "const", "glorot", "ortho", "gaussian"], "Unknown init method %s" % init
            if len(shape) == 1:     # set bias to 0
                return dy.ConstInitializer(0.)
            elif len(shape) == 2:
                if init == "default" or init == "glorot":
                    return dy.GlorotInitializer()
                elif init == "gaussian":
                    return dy.NormalInitializer(var=0.01*0.01)
                elif init == "ortho":
                    assert shape[0]%shape[1] == 0, "Bad shape %s for ortho_init" % shape
                    num = shape[0] // shape[1]
                    arr = ortho_weight(shape[1]) if num == 1 else\
                          np.concatenate([ortho_weight(shape[1]) for _ in range(num)])
                    return dy.NumpyInitializer(arr)
            else:
                raise NotImplementedError("Currently only support parameter dim <= 2.")
        # first, if init is np-array
        if isinstance(init, np.ndarray):
            return self.model.add_parameters(shape, init=dy.NumpyInitializer(init))
        # then ...
        if lookup:
            return self.model.add_lookup_parameters(shape)  # also default Glorot
        # shape is a tuple of dims
        if len(shape) == 1:     # set bias to 0
            return self.model.add_parameters(shape, init=dy.ConstInitializer(0.))
        else:
            return self.model.add_parameters(shape, init=get_init(shape, init))

# linear layer with selectable activation functions
class Linear(Basic):
    def __init__(self, model, n_in, n_out, act="tanh"):
        super(Linear, self).__init__(model)
        self.params["W"] = self._add_parameters((n_out, n_in))
        self.params["B"] = self._add_parameters((n_out,))
        self.act = {"tanh":dy.tanh, "softmax":dy.softmax, "linear":None}[act]

    def refresh(self, argv):
        self._ingraph(argv)

    def __call__(self, input_exp):
        x = dy.affine_transform([self.iparams["B"], self.iparams["W"], input_exp])
        if self.act is not None:
            x = self.act(x)
        if self.hdrop > 0.:
            x = dy.dropout(x, self.hdrop)
        return x

# linear layer with multi inputs
class LinearMulti(Basic):
    def __init__(self, model, n_ins, n_out, act="tanh"):
        super(LinearMulti, self).__init__(model)
        for i, n_in in enumerate(n_ins):
            self.params["W%s"%i] = self._add_parameters((n_out, n_in))
        self.params["B"] = self._add_parameters((n_out,))
        self.act = {"tanh":dy.tanh, "softmax":dy.softmax, "linear":None}[act]
        self._pattern = []
        self.n_num = len(n_ins)

    def refresh(self, argv):
        self._ingraph(argv)
        # prepare calculation list
        self._pattern = [self.iparams["B"]]
        for i in range(self.n_num):
            self._pattern += [self.iparams["W%s"%i], None]

    def __call__(self, input_exps):
        assert len(input_exps) == self.n_num
        for i in range(self.n_num):
            self._pattern[2*i+2] = input_exps[i]
        x = dy.affine_transform(self._pattern)
        if self.act is not None:
            x = self.act(x)
        if self.hdrop > 0.:
            x = dy.dropout(x, self.hdrop)
        return x

# embedding layer
class Embedding(Basic):
    def __init__(self, model, n_words, n_dim, dropout_wordceil=None, npvec=None):
        super(Embedding, self).__init__(model)
        if npvec is not None:
            assert len(npvec.shape) == 2 and npvec.shape[0] == n_words and npvec.shape[1] == n_dim
            self.params["_E"] = self.model.lookup_parameters_from_numpy(npvec)
        else:
            self.params["_E"] = self._add_parameters((n_words, n_dim), lookup=True)
        self.n_dim = n_dim
        self.n_words = n_words
        self.dropout_wordceil = dropout_wordceil if dropout_wordceil is not None else n_words

    def refresh(self, argv):
        # zero out
        self.params["_E"].init_row(0, [0. for _ in range(self.n_dim)])
        # refresh
        self._ingraph(argv)

    def __call__(self, input_exp):
        # input should be a list of ints, masks should be a set of ints for the dropped ints
        if type(input_exp) != list:
            input_exp = [input_exp]
        # input dropouts
        if self.idrop > 0:
            input_exp = [(0 if (v>0.5 and i<self.dropout_wordceil) else i) for i,v in zip(input_exp, np.random.binomial(1, self.idrop, len(input_exp)))]
        x = dy.lookup_batch(self.params["_E"], input_exp, self.update)
        if self.hdrop > 0:
            x = dy.dropout(x, self.hdrop)
        return x

# rnn nodes
class RnnNode(Basic):
    def __init__(self, model, n_input, n_hidden):
        super(RnnNode, self).__init__(model)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.width = -1

    def refresh(self, argv):
        # refresh
        self._ingraph(argv)
        if self.gdrop > 0:   # ensure same masks for all instances in the batch
            # todo: 1. gdrop for both or rec-only? 2. diff gdrop for gates or not? 3. diff for batches or not
            # special for the current two: [xgate, hgate, xinput, hinput]
            self.masks = [gen_masks_input(self.gdrop, self.n_input, 1), gen_masks_input(self.gdrop, self.n_hidden, 1),
                          gen_masks_input(self.gdrop, self.n_input, 1), gen_masks_input(self.gdrop, self.n_hidden, 1),]

    def __call__(self, input_exp, hidden_exp, mask=None):
        # todo(warn) return a {}
        raise NotImplementedError()

    @staticmethod
    def get_rnode(s):   # todo: gru2
        return {"gru": GruNode, "gru2": GruNode2}[s]

class GruNode(RnnNode):
    def __init__(self, model, n_input, n_hidden):
        super(GruNode, self).__init__(model, n_input, n_hidden)
        # paramters
        self.params["x2r"] = self._add_parameters((n_hidden, n_input))
        self.params["h2r"] = self._add_parameters((n_hidden, n_hidden), init="ortho")
        self.params["br"] = self._add_parameters((n_hidden,))
        self.params["x2z"] = self._add_parameters((n_hidden, n_input))
        self.params["h2z"] = self._add_parameters((n_hidden, n_hidden), init="ortho")
        self.params["bz"] = self._add_parameters((n_hidden,))
        self.params["x2h"] = self._add_parameters((n_hidden, n_input))
        self.params["h2h"] = self._add_parameters((n_hidden, n_hidden), init="ortho")
        self.params["bh"] = self._add_parameters((n_hidden,))

    def __call__(self, input_exp, hidden_exp, mask=None):
        # two kinds of dropouts
        if self.idrop > 0.:
            input_exp = dy.dropout(input_exp, self.idrop)
        input_exp_g = input_exp_t = input_exp
        hidden_exp_g = hidden_exp_t = hidden_exp["H"]
        if self.gdrop > 0.:
            input_exp_g = dy.cmult(input_exp, self.masks[0])
            hidden_exp_g = dy.cmult(hidden_exp_g, self.masks[1])
            input_exp_t = dy.cmult(input_exp, self.masks[2])
            hidden_exp_t = dy.cmult(hidden_exp_t, self.masks[3])
        rt = dy.affine_transform([self.iparams["br"], self.iparams["x2r"], input_exp_g, self.iparams["h2r"], hidden_exp_g])
        rt = dy.logistic(rt)
        zt = dy.affine_transform([self.iparams["bz"], self.iparams["x2z"], input_exp_g, self.iparams["h2z"], hidden_exp_g])
        zt = dy.logistic(zt)
        h_reset = dy.cmult(rt, hidden_exp_t)
        ht = dy.affine_transform([self.iparams["bh"], self.iparams["x2h"], input_exp_t, self.iparams["h2h"], h_reset])
        ht = dy.tanh(ht)
        hidden = dy.cmult(zt, hidden_exp["H"]) + dy.cmult((1. - zt), ht)     # first one use original hh
        # mask: if 0 then pass through
        if mask is not None:
            mask_array = np.asarray(mask).reshape((1, -1))
            m1 = dy.inputTensor(mask_array, True)           # 1.0 for real words
            m0 = dy.inputTensor(1.0 - mask_array, True)     # 1.0 for padding words (mask=0)
            hidden = hidden * m1 + hidden_exp["H"] * m0
        return {"H": hidden}

class GruNode2(RnnNode):
    def __init__(self, model, n_input, n_hidden):
        super(GruNode2, self).__init__(model, n_input, n_hidden)
        # paramters
        self.params["x2rz"] = self._add_parameters((2*n_hidden, n_input))
        self.params["h2rz"] = self._add_parameters((2*n_hidden, n_hidden), init="ortho")
        self.params["brz"] = self._add_parameters((2*n_hidden,))
        self.params["x2h"] = self._add_parameters((n_hidden, n_input))
        self.params["h2h"] = self._add_parameters((n_hidden, n_hidden), init="ortho")
        self.params["bh"] = self._add_parameters((n_hidden,))

    def __call__(self, input_exp, hidden_exp, mask=None):
        # two kinds of dropouts
        if self.idrop > 0.:
            input_exp = dy.dropout(input_exp, self.idrop)
        input_exp_g = input_exp_t = input_exp
        hidden_exp_g = hidden_exp_t = hidden_exp["H"]
        if self.gdrop > 0.:
            input_exp_g = dy.cmult(input_exp, self.masks[0])
            hidden_exp_g = dy.cmult(hidden_exp_g, self.masks[1])
            input_exp_t = dy.cmult(input_exp, self.masks[2])
            hidden_exp_t = dy.cmult(hidden_exp_t, self.masks[3])
        rzt = dy.affine_transform([self.iparams["brz"], self.iparams["x2rz"], input_exp_g, self.iparams["h2rz"], hidden_exp_g])
        rzt = dy.logistic(rzt)
        rt, zt = dy.pick_range(rzt, 0, self.n_hidden), BK.pick_range(rzt, self.n_hidden, 2*self.n_hidden)
        h_reset = dy.cmult(rt, hidden_exp_t)
        ht = dy.affine_transform([self.iparams["bh"], self.iparams["x2h"], input_exp_t, self.iparams["h2h"], h_reset])
        ht = dy.tanh(ht)
        hidden = dy.cmult(zt, hidden_exp["H"]) + dy.cmult((1. - zt), ht)     # first one use original hh
        # mask: if 0 then pass through
        if mask is not None:
            mask_array = np.asarray(mask).reshape((1, -1))
            m1 = dy.inputTensor(mask_array, True)           # 1.0 for real words
            m0 = dy.inputTensor(1.0 - mask_array, True)     # 1.0 for padding words (mask=0)
            hidden = hidden * m1 + hidden_exp["H"] * m0
        return {"H": hidden}

class Attention(Basic):
    def __init__(self, model, n_s, n_h, n_hidden):
        super(Attention, self).__init__(model)
        # info
        self.n_s, self.n_h, self.n_hidden = n_s, n_h, n_hidden

    def refresh(self, argv):
        self._ingraph(argv)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("No calling __call__ from Attention.")

    def _restart_caches(self, sent, c):
        raise NotImplementedError("No calling __call__ from Attention.")

    @staticmethod
    def get_attentioner(s):
        return {"ff":FfAttention, "biaff":BiaffAttention}[s]

# feed forward for attention --- requiring much memory
class FfAttention(Attention):
    def __init__(self, model, n_s, n_h, n_hidden):
        super(FfAttention, self).__init__(model, n_s, n_h, n_hidden)
        # parameters -- (feed-forward version)
        self.params["s2e"] = self._add_parameters((n_hidden, n_s))
        self.params["h2e"] = self._add_parameters((n_hidden, n_h))
        self.params["v"] = self._add_parameters((1, n_hidden))

    def _restart_caches(self, sent, c):
        if c is None:
            caches = {}
            caches["S"] = dy.concatenate_cols(sent)
            caches["V"] = self.iparams["s2e"] * caches["S"]
        else:
            caches = {"S":c["S"], "V":c["V"]}
        return caches

    def __call__(self, sent, n, caches):
        # s: list(len==steps) of {(n_s,), batch_size}, n: {(n_h,), batch_size}
        caches = self._restart_caches(sent, caches)
        val_h = self.iparams["h2e"] * n     # {(n_hidden,), batch_size}
        att_hidden_bef = dy.colwise_add(caches["V"], val_h)    # {(n_didden, steps), batch_size}
        att_hidden = dy.tanh(att_hidden_bef)
        # if self.hdrop > 0:     # save some space
        #     att_hidden = dy.dropout(att_hidden, self.hdrop)
        att_e = dy.reshape(self.iparams["v"] * att_hidden, (BK.dims(caches["V"])[1], ), batch_size=bs(att_hidden))
        att_alpha = dy.softmax(att_e)
        ctx = caches["S"] * att_alpha      # {(n_s, sent_len), batch_size}
        # append and return
        caches["ctx"] = ctx
        caches["att"] = att_alpha
        return caches

class BiaffAttention(Attention):
    def __init__(self, model, n_s, n_h, n_hidden):
        super(BiaffAttention, self).__init__(model, n_s, n_h, n_hidden)
        # parameters -- (BiAffine-version e = h*W*s)
        self.params["W"] = self._add_parameters((n_h, n_s))

    def _restart_caches(self, sent, c):
        if c is None:
            caches = {}
            caches["S"] = dy.concatenate_cols(sent)
            caches["V"] = self.iparams["W"] * caches["S"]
        else:
            caches = {"S":c["S"], "V":c["V"]}
        return caches

    def __call__(self, sent, n, caches):
        caches = self._restart_caches(sent, caches)
        # s: list(len==steps) of {(n_s,), batch_size}, n: {(n_h,), batch_size}
        wn_t = dy.reshape(n, (1, self.n_h), batch_size=bs(n))
        att_e = dy.reshape(wn_t * caches["V"], (BK.dims(caches["V"])[1], ), batch_size=bs(n))
        att_alpha = dy.softmax(att_e)
        ctx = caches["S"] * att_alpha
        # append and return
        caches["ctx"] = ctx
        caches["att"] = att_alpha
        return caches

# ================= Blocks ================= #
# stateless encoder
class Encoder(object):
    def __init__(self, model, n_input, n_hidden, n_layers, rnn_type):
        # [[f,b], ...]
        self.ntype = RnnNode.get_rnode(rnn_type)
        self.nodes = [[self.ntype(model, n_input, n_hidden), self.ntype(model, n_input, n_hidden)]]
        for i in range(n_layers-1):
            self.nodes.append([self.ntype(model, n_hidden, n_hidden), self.ntype(model, n_hidden, n_hidden)])
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers

    def refresh(self, argv):
        for nn in self.nodes:
            nn[0].refresh(argv)
            nn[1].refresh(argv)

    def __call__(self, embeds, masks):
        # embeds: list(step) of {(n_emb, ), batch_size}, using padding for batches
        b_size = bs(embeds[0])
        outputs = [embeds]
        # # todo(warn), disable masks for speeding up (although might not be critical)
        # masks = [None for _ in masks]
        for i, nn in zip(range(self.n_layers), self.nodes):
            init_hidden = dy.zeroes((self.n_hidden,), batch_size=b_size)
            tmp_f = []      # forward
            tmp_f_prev = {"H":init_hidden, "C":init_hidden}
            for e, m in zip(outputs[-1], masks):
                one_output = nn[0](e, tmp_f_prev, m)
                tmp_f.append(one_output["H"])
                tmp_f_prev = one_output
            tmp_b = []      # forward
            tmp_b_prev = {"H":init_hidden, "C":init_hidden}
            for e, m in zip(reversed(outputs[-1]), reversed(masks)):
                one_output = nn[1](e, tmp_b_prev, m)
                tmp_b.append(one_output["H"])
                tmp_b_prev = one_output
            # concat
            ctx = [dy.concatenate([f,b]) for f,b in zip(tmp_f, reversed(tmp_b))]
            outputs.append(ctx)
        return outputs[-1]

# attentional decoder
class Decoder(object):
    def __init__(self, model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden, att_type, rnn_type, summ_type):
        self.ntype = RnnNode.get_rnode(rnn_type)
        self.all_nodes = []
        # gru nodes --- wait for the sub-classes
        # init nodes
        self.inodes = [Linear(model, dim_src, n_hidden, act="tanh") for _ in range(n_layers)]
        for inod in self.inodes:
            self.all_nodes.append(inod)
        # att node
        self.anode = Attention.get_attentioner(att_type)(model, dim_src, n_hidden, dim_att_hidden)
        self.all_nodes.append(self.anode)
        # info
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dim_src = dim_src      # also the size of attention vector
        # summarize for the source as the start of decoder
        self.summer = Decoder.get_summer(summ_type, dim_src)    # bidirection

    @staticmethod
    def get_summer(s, size):  # list of values (bidirection) => one value
        if s == "avg":
            return dy.average
        else:
            mask = [0. for _ in range(size//2)]+[1. for _ in range(size//2)]
            mask2 = [1. for _ in range(size//2)]+[0. for _ in range(size//2)]
            if s == "fend":
                return lambda x: dy.cmult(dy.inputVector(mask2), x[-1])
            elif s == "bend":
                return lambda x: dy.cmult(dy.inputVector(mask), x[0])
            elif s == "ends":
                return lambda x: dy.cmult(dy.inputVector(mask2), x[-1]) + dy.cmult(dy.inputVector(mask), x[0])
            else:
                return None

    def refresh(self, argv):
        for nn in self.all_nodes:
            nn.refresh(argv)    # dropouts: init/att: hdrop, rec: idrop, gdrop

    def start_one(self, s):
        # start to decode with one sentence, W*summ(s) as init
        # also expand on the batch dimension # TODO: maybe should put at the col dimension
        inits = []
        summ = self.summer(s)
        for i in range(self.n_layers):
            cur_init = self.inodes[i](summ)
            # +1 for the init state
            inits.append({"H": cur_init})
        caches = self.anode(s, inits[0]["H"], None)          # start of the attention
        # append and return
        caches["summ"] = summ
        caches["hid"] = inits
        return caches

    def feed_one(self, s, inputs, caches, prev_embeds=None):
        assert bs(caches["hid"][0]["H"]) == bs(inputs), "Unmatched batch_size"
        return self._feed_one(s, inputs, caches, prev_embeds)

    def _feed_one(self, s, inputs, caches, prev_embeds):
        raise NotImplementedError("Decoder should be inherited!")

# normal attention decoder
class AttDecoder(Decoder):
    def __init__(self, model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden, att_type, rnn_type, summ_type):
        super(AttDecoder, self).__init__(model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden, att_type, rnn_type, summ_type)
        # gru nodes
        self.gnodes = [self.ntype(model, n_input+dim_src, n_hidden)]    # (E(y_{i-1})//c_i, s_{i-1}) => s_i
        for i in range(n_layers-1):
            self.gnodes.append(self.ntype(model, n_hidden, n_hidden))
        for gnod in self.gnodes:
            self.all_nodes.append(gnod)

    def _feed_one(self, s, inputs, caches, prev_embeds):
        # first layer with attetion
        next_caches = self.anode(s, caches["hid"][0]["H"], caches)
        g_input = dy.concatenate([inputs, next_caches["ctx"]])
        hidd = self.gnodes[0](g_input, caches["hid"][0])
        this_hiddens = [hidd]
        # later layers
        for i in range(1, self.n_layers):
            ihidd = self.gnodes[i](this_hiddens[i-1]["H"], caches["hid"][i])
            this_hiddens.append(ihidd)
        # append and return
        next_caches["hid"] = this_hiddens
        return next_caches

# nematus-style attention decoder, fixed two transitions
class NematusDecoder(Decoder):
    def __init__(self, model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden, att_type, rnn_type, summ_type):
        super(NematusDecoder, self).__init__(model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden, att_type, rnn_type, summ_type)
        # gru nodes
        self.gnodes = [self.ntype(model, n_input, n_hidden)]        # gru1 for the first layer
        for i in range(n_layers-1):
            self.gnodes.append(self.ntype(model, n_hidden, n_hidden))
        self.gnodes.append(self.ntype(model, dim_src, n_hidden))   # gru2 for the first layer
        for gnod in self.gnodes:
            self.all_nodes.append(gnod)

    def _feed_one(self, s, inputs, caches, prev_embeds):
        # first layer with attetion, gru1 -> att -> gru2
        s1 = self.gnodes[0](inputs, caches["hid"][0])
        next_caches = self.anode(s, s1["H"], caches)
        hidd = self.gnodes[-1](next_caches["ctx"], s1)
        this_hiddens = [hidd]
        # later layers
        for i in range(1, self.n_layers):
            ihidd = self.gnodes[i](this_hiddens[i-1]["H"], caches["hid"][i])
            this_hiddens.append(ihidd)
        # append and return
        next_caches["hid"] = this_hiddens
        return next_caches

# as the name suggests, like a feed-forward NN
class NgramDecoder(Decoder):
    def __init__(self, model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden, att_type, rnn_type, summ_type, ngram):
        super(NgramDecoder, self).__init__(model, n_input, n_hidden, n_layers, dim_src, dim_att_hidden, att_type, rnn_type, summ_type)
        self.ngram = ngram      # todo(warn) include the most recent one, with redundancy but for attention
        self.layers = [LinearMulti(model, [n_input]*ngram, n_hidden)]
        for i in range(n_layers-1):
            self.layers.append(Linear(model, n_hidden, n_hidden))
        self.all_nodes += self.layers

    def _feed_one(self, s, inputs, caches, prev_embeds):
        # todo(warn) prev_embeds are prepared by up-layers and only utilized here in Ngram-Decoder,
        # -- attention not in the calculation of "hid"
        # ignore previous hidden & inputs
        ihidd = self.layers[0](prev_embeds)
        for i in range(1, self.n_layers):
            ihidd = self.layers[i](ihidd)
        next_caches = self.anode(s, ihidd, caches)
        # append and return
        next_caches["hid"] = [{'H': ihidd}]      # todo(warn): ugly
        return next_caches

    def refresh(self, argv):
        super(NgramDecoder, self).refresh(argv)
        # todo(warn): special setting with gdrop to make it similar to RnnDecoder
        for one in self.layers:
            one.hdrop = max(one.hdrop, one.gdrop)
