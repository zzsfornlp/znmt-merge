# about the length of output sentence
import numpy as np
from zl import utils
try:
    from sklearn import linear_model
except:
    pass

from .mt_layers import Basic as BasicLayer
from zl.layers import BK

# ================ for training
# how to take lengths into account (simple scale)
class MTLengther(object):
    @staticmethod
    def get_scaler_f(method, alpha):
        if method == "none":
            return None
        if alpha <= 0.:
            return (lambda l: 1.)
        else:
            return {"norm": (lambda l: 1. / pow(l, alpha)),
                    "google": (lambda l: 1. * pow(6, alpha) / pow(5+l, alpha))}[method]

class LinearGaussain(BasicLayer):
    SCALE_LENGTH = 100.0
    INIT_WIDTH = 5
    _DEFAULT_INFO = (1., 0., INIT_WIDTH/SCALE_LENGTH, 0.)

    @staticmethod
    def trans_len(l):
        return l / LinearGaussain.SCALE_LENGTH

    @staticmethod
    def back_len(l):
        return l * LinearGaussain.SCALE_LENGTH

    @staticmethod
    def log_gaussian(one, mu, si):
        _c = si ** 2
        # r = - (one-mu)**2/(2*_c) - 0.5*np.log(2*np.pi*_c)
        r = - (one-mu)**2/(2*_c)
        return r

    @staticmethod
    def fit_once(fit_files):
        # first fitting a simple one: y = gaussian(ax+b, sigma), here not including xenc for that will be too large
        with utils.Timer(tag="Fit-length-once", print_date=True):
            # 1. collect length
            with utils.zopen(fit_files[0]) as f0, utils.zopen(fit_files[1]) as f1:
                # todo(warn): plus one for the <eos> tokens
                x = [LinearGaussain.trans_len(len(_l.split())+1) for _l in f0]
                y = [LinearGaussain.trans_len(len(_l.split())+1) for _l in f1]
            utils.zcheck_matched_length(x, y, _forced=True)
            ll = len(x)
            x1, y1 = np.array(x, dtype=np.float32).reshape((-1,1)), np.array(y, dtype=np.float32)
            # 2. fit linear model
            try:    # todo(warn)
                regr = linear_model.LinearRegression()
                regr.fit(x1, y1)
                a, b = float(regr.coef_[0]), float(regr.intercept_)
            except:
                utils.zlog("Cannot linear-regression, skip that.")
                a, b = 1., 0.
            # 3. fit sigma
            x1 = x1.reshape((-1,))
            errors = a*x1+b - y1
            mu = np.mean(errors)
            sigma = np.sqrt(((errors - mu)**2).mean())
            ret = (a, b, sigma, mu)
            del x, y, x1, y1
            utils.zlog("Fitting Length with %s sentences and get %s." % (ll, ret), func="score")
        return ret

    def __init__(self, model, xlen, xadd, xback, length_info):
        super(LinearGaussain, self).__init__(model)
        self.xlen = xlen        # dim of src repr
        self.xadd = xadd        # whether add xsrc
        self.xback = xback      # whether prop back through xsrc
        if length_info is None:
            length_info = LinearGaussain._DEFAULT_INFO
        # params
        utils.zlog("Init lg with len-info %s" % (length_info,))
        self.params["W"] = self._add_params((1, xlen), )
        self.params["A"] = self._add_params((1, 1), init=np.array([length_info[0],], dtype=np.float32))
        self.params["B"] = self._add_params((1,), init=np.array([length_info[1],], dtype=np.float32))
        self.params["SI"] = self._add_params((1,), init=np.array([length_info[2],], dtype=np.float32))

    def refresh(self, argv):
        self._ingraph(argv)

    # def __call__(self, , xsrc, xlens):
    #     return self.calculate(xsrc, xlens)

    def calculate(self, xsrc, xlens):
        # xsrc is Expr(batched), xlens is list-int
        xx = np.array(xlens, dtype=np.float32).reshape((1,-1))
        xx = LinearGaussain.trans_len(xx)
        xe = BK.inputTensor(xx, True)
        utils.zcheck(BK.bsize(xe) == BK.bsize(xsrc), "Unmatched batch-size!")
        # forward
        input_list = [self.iparams["B"], self.iparams["A"], xe]
        if self.xadd:
            if self.idrop > 0.:
                xsrc = BK.dropout(xsrc, self.idrop)
            if self.xback:
                source = xsrc
            else:
                source = BK.nobackprop(xsrc)
            input_list += [self.iparams["W"], source]
        pred_lens = BK.affine(input_list)
        return pred_lens

    def ll_loss(self, pred_lens, ylens):
        xx = np.array(ylens, dtype=np.float32).reshape((1,-1))
        xx = LinearGaussain.trans_len(xx)
        xe = BK.inputTensor(xx, True)
        utils.zcheck(BK.bsize(xe) == BK.bsize(pred_lens), "Unmatched batch-size!")
        # loss = -log(Norm)/bsize = log(SI) + (\Delta^2)/2*sig*2
        loss = BK.log(self.iparams["SI"]) + BK.cdiv(BK.mean_batches(BK.square(pred_lens-xe)), (BK.square(self.iparams["SI"]) * 2))
        return loss

    def obtain_params(self):
        a,b,sigma = BK.param2np(self.params["A"]), BK.param2np(self.params["B"]), BK.param2np(self.params["SI"])
        return a,b,sigma

    def get_real_sigma(self):
        return LinearGaussain.back_len(BK.param2np(self.params["SI"])[0])

# =============== for testing
# testing time or training a length fitter
# todo(warn): deeply coupled with State
def get_normer(method, alpha, gaussian_sigma):
    if method in ["none", "norm", "google"]:
        return ScaleLengthNormer(method, alpha)
    elif method == "add":
        return AdderNormer(alpha, gaussian_sigma)
    else:
        assert method in ["gaussian", "xgaussian"]
        return GaussianNormer(alpha, gaussian_sigma)

class ScaleLengthNormer(object):
    def __init__(self, method, alpha):
        self._ff = MTLengther.get_scaler_f(method, alpha)
        if self._ff is None:
            self._ff = MTLengther.get_scaler_f("norm", -1)      # for convenience

    def __call__(self, states, pad):
        for beam in states:
            for one in beam:
                one.set_score_final(one.score_partial * self._ff(one.length))

class AdderNormer(object):
    def __init__(self, alpha, sigma):
        self._alpha = alpha
        self._sigma = sigma

    def __call__(self, states, lengths):
        utils.zcheck_matched_length(states, lengths, _forced=True)
        for beam, _l in zip(states, lengths):
            for one in beam:
                # todo(warn): mu+2*si -> 95%
                el = min(one.length, _l+2*self._sigma)
                one.set_score_final(one.score_partial + el*self._alpha)

class GaussianNormer(object):
    def __init__(self, alpha, sigma):
        self._alpha = alpha
        self._sigma = sigma

    def __call__(self, states, lengths):
        utils.zcheck_matched_length(states, lengths, _forced=True)
        for beam, _l in zip(states, lengths):
            for one in beam:
                log_prob = LinearGaussain.log_gaussian(one.length, _l, self._sigma)
                log_prob *= self._alpha
                one.set_score_final(one.score_partial + log_prob)
