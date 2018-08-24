from collections import Iterable
from .common import *
import sys

import _dynet as dy

affine = dy.affine_transform
average = concat_wrapper(dy.average)
cmult = dy.cmult
cdiv = dy.cdiv
colwise_add = dy.colwise_add
concatenate = concat_wrapper(dy.concatenate)
concatenate_cols = concat_wrapper(dy.concatenate_cols)
concatenate_to_batch = concat_wrapper(dy.concatenate_to_batch)
dropout = dy.dropout
esum = dy.esum
log = dy.log
hinge_batch = dy.hinge_batch
inputTensor = dy.inputTensor
inputVector = dy.inputVector
inputVector_batch = lambda x: reshape(inputVector(x), (1,), len(x))
logistic = dy.logistic
lookup_batch = dy.lookup_batch
max_dim = dy.max_dim
mean_batches = dy.mean_batches
nobackprop = dy.nobackprop
pickneglogsoftmax_batch = dy.pickneglogsoftmax_batch
pick_batch_elems = dy.pick_batch_elems
pick_batch_elem = dy.pick_batch_elem
pick_batch = dy.pick_batch
pick_range = dy.pick_range
rectify = dy.rectify
reshape = dy.reshape
softmax = dy.softmax
square = dy.square
sum_batches = dy.sum_batches
sum_elems = dy.sum_elems
tanh = dy.tanh
transpose = dy.transpose
zeros = dy.zeros

def random_bernoulli(rate, size, bsize):
    return dy.random_bernoulli((size,), 1.-rate, 1./(1.-rate), batch_size=bsize)

def new_graph():
    # dy.renew_cg(immediate_compute = True, check_validity = True)   # new graph
    dy.renew_cg(immediate_compute=DY_CONFIG.immediate_compute)

def new_model():
    return dy.ParameterCollection()

def load_model(fname, m):
    m.populate(fname)
    return m

def save_model(fname, m):
    m.save(fname)

def param2expr(p, update):
    # todo(warn): dynet changes API
    if not update:
        raise NotImplementedError("No longer specified here.")
    # try:
    #     e = dy.parameter(p, update)
    # except NotImplementedError:
    #     if update:
    #         e = dy.parameter(p)
    #     else:
    #         e = dy.const_parameter(p)
    return p

def param2np(p):
    return p.as_array()

def forward(expr):
    expr.forward()

def backward(expr):
    expr.backward()

def get_value_vec(expr):
    expr.forward()
    return expr.vec_value()

def get_value_sca(expr):
    expr.forward()
    return expr.scalar_value()

def get_value_np(expr):
    v = get_value_vec(expr)
    shape = [bsize(expr)] + list(reversed(dims(expr)))
    return np.asarray(v).reshape(shape)

# def get_value_np(expr):
#     expr.forward()
#     return expr.npvalue()

# def get_value_v(expr):
#     data = get_value_vec(expr)
#     return Value(data, list(reversed(dims(expr))), bsize(expr))

def get_params(model, shape, lookup=False, init="default"):
    if isinstance(init, np.ndarray):    # pass it directly
        arr = init
    else:
        arr = get_params_init(shape, init, lookup)
    if lookup:
        p = model.lookup_parameters_from_numpy(arr)
    else:
        p = model.add_parameters(shape, init=dy.NumpyInitializer(arr))
    return p

def gru():
    pass

def vanilla_lstm(iis, hh, cc, px, ph, b, dropx, droph):
    if dropx is not None and droph is not None:
        gates_t = dy.vanilla_lstm_gates_concat_dropout(iis, hh, px, ph, b, dropx, droph)
    else:
        gates_t = dy.vanilla_lstm_gates_concat(iis, hh, px, ph, b)
    cc = dy.vanilla_lstm_c(cc, gates_t)
    hidden = dy.vanilla_lstm_h(cc, gates_t)
    return hidden, cc

def dims(expr):
    return expr.dim()[0]

def bsize(expr):
    return expr.dim()[-1]

# manipulating the batches #

# def batch_rearrange(exprs, orders):
#     if not isinstance(exprs, Iterable):
#         exprs = [exprs]
#     if not isinstance(orders, Iterable):
#         orders = [orders]
#     utils.zcheck_matched_length(exprs, orders, _forced=True)
#     new_ones = []
#     for e, o in zip(exprs, orders):
#         new_ones.append(pick_batch_elems(e, o))
#     return concatenate_to_batch(new_ones)

def batch_rearrange_one(e, o):
    return pick_batch_elems(e, o)

def batch_repeat(expr, num=1):
    # repeat each element (expanding) in the batch
    utils.zcheck_range(num, 1, None, _forced=True)
    if num == 1:
        return expr
    else:
        bs = bsize(expr)
        orders = [i//num for i in range(bs*num)]
        return batch_rearrange_one(expr, orders)

class Trainer(object):
    def __init__(self, model, type, lrate, moment=None):
        self._tt = {"sgd": dy.SimpleSGDTrainer(model, lrate),
                        "momentum": dy.MomentumSGDTrainer(model, lrate, moment),
                        "adam": dy.AdamTrainer(model, lrate)
                        }[type]

    def restart(self):
        self._tt.restart()

    def set_lrate(self, lr):
        self._tt.learning_rate = lr

    def set_clip(self, cl):
        self._tt.set_clip_threshold(cl)

    def save_shadow(self, fname):
        # todo
        utils.zcheck(False, "Not implemented for saving shadows.", func="warn")

    def load_shadow(self, fname):
        # todo
        utils.zcheck(False, "Not implemented for loading shadows.", func="warn")

    def update(self):
        self._tt.update()

def rearrange_cache(cache, order):
    if isinstance(cache, dict):
        ret = {}
        for n in cache:
            ret[n] = rearrange_cache(cache[n], order)
        return ret
    elif isinstance(cache, list):
        return [rearrange_cache(_i, order) for _i in cache]
    elif isinstance(cache, type(None)):
        return None
    else:
        return batch_rearrange_one(cache, order)

def recombine_cache(caches, indexes):
    # input lists, output combine one. todo: to be more efficient
    c0 = caches[0]
    if isinstance(c0, dict):
        ret = {}
        for n in c0:
            ret[n] = recombine_cache([_c[n] for _c in caches], indexes)
        return ret
    elif isinstance(c0, list):
        return [recombine_cache([_c[_i] for _c in caches], indexes) for _i in range(len(c0))]
    elif isinstance(c0, type(None)):
        return None
    else:
        them = [pick_batch_elem(_c, _i) for _c, _i in zip(caches, indexes)]
        return concatenate_to_batch(them)

# ------
def add_margin(bexpr, yidxs, margin):
    dd, bs = dims(bexpr), bsize(bexpr)
    utils.zcheck(bs == len(yidxs), "Unmatched bsize for margin adding.")
    utils.zcheck(len(dd) == 1, "Currently only support dim=1.")
    # reverse, adding margin except the correct ones
    adding = dy.sparse_inputTensor((yidxs, [i for i in range(bs)]), [0. for _ in yidxs], shape=(dd[0], bs), batched=True, defval=margin)
    bexpr2 = bexpr + adding
    return bexpr2

# def count_nonzero(expr):
#     TMP_MUL = 1000
#     dd, bs = dims(expr), bsize(expr)
#     utils.zcheck(len(dd) == 1, "Currently only support dim=1.")
#     thresh_expr = dy.constant((dd[0],), 1/TMP_MUL, batch_size=bs)
#     min_expr = dy.bmin(expr, thresh_expr)
#     count_expr = dy.sum_elems(min_expr) * TMP_MUL
#     return nobackprop(count_expr)

# def topk(expr, k):
#     assert k==1, "only support k==1"
#     tv = expr.tensor_value()
#     max_idx = tv.argmax()
#     idxs = max_idx.as_numpy()
#     max_exprs = pick_batch(expr, idxs[0])
#     max_val = get_value_vec(max_exprs)
#     return idxs, max_val

# =============== topk
def NEVER_KNOWN_LOCAL_nargmax(v, n):
    # return ORDERED list of (id, value)
    thres = max(-len(v), -n)
    ids = np.argpartition(v, thres)[thres:]
    ret = sorted([int(i) for i in ids], key=lambda x: v[x], reverse=True)
    return ret

def topk_cpu(expr, k, prepare=True):
    value_np = get_value_np(expr)
    idxs, vals = [], []
    for vip in value_np:
        one_idxs = NEVER_KNOWN_LOCAL_nargmax(vip, k)
        one_vals = vip[one_idxs]
        idxs += one_idxs
        vals += one_vals.tolist()
    pp = (idxs, vals)
    if prepare:
        return ResultTopk.prepare_results(pp, k)
    else:
        return pp

def topk_gpu(expr, k, prepare=True):
    # get the k-argmax_and_max of an expr, return a list(BATCH) of list(K) of pairs
    # -- only for batched-dim0
    # todo(warn): guarantee sorted
    tv = expr.tensor_value()
    # pp = tv.max_and_argmax(0, k)
    vals, idxes = tv.topk(0, k)
    pp = (idxes.as_numpy().T.flatten(), vals.as_numpy().T.flatten())
    # debug
    # if True:
    #     ppc = topk_cpu(expr, k, False)
    #     utils.zcheck(ppc[0]==pp[0], "Error on topk.")
    if prepare:
        return ResultTopk.prepare_results(pp, k)
    else:
        return pp

topk = topk_gpu

# =============== count_larger
def cl_cpu(ex, ey):
    vx = get_value_np(ex)
    vy = get_value_np(ey)
    ret = []
    for i in range(len(vx)):
        c = np.sum(vx[i]>vy[i])
        ret.append(int(c))
    return ret

def cl_gpu(ex, ey):
    tv = ex.tensor_value()
    ty = ey.tensor_value()
    pp = tv.count_larger(ty)
    # # debug
    # if True:
    #     ppc = cl_cpu(ex, ey)
    #     utils.zcheck(ppc==pp, "Error on count_larger.")
    return pp

count_larger = cl_gpu

# ------------- CONFIGS -------------
class DY_CONFIG:
    immediate_compute = False

def init(opts):
    # todo: manipulating sys.argv
    utils.zlog("Using BACKEND of DYNET on %s." % (opts["dynet-devices"],))
    params = dy.DynetParams()
    temp = sys.argv
    sys.argv = [temp[0], "--dynet-mem", opts["dynet-mem"], "--dynet-autobatch", opts["dynet-autobatch"],
                "--dynet-devices", opts["dynet-devices"], "--dynet-seed", opts["dynet-seed"]]
    DY_CONFIG.immediate_compute = opts["dynet-immed"]
    params.from_args(None)
    params.init()
    sys.argv = temp
    if "GPU" not in opts["dynet-devices"]:
        global topk
        topk = topk_cpu
        global count_larger
        count_larger = cl_cpu
        utils.zlog("Currently using numpy for topk_cpu/count_larger.")
