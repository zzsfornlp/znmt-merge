# the model which contains the params and could be used to score candidates

from . import layers, utils
import json

class Model(object):
    PROP_SUFFIX = ".prop"

    def __init__(self):
        utils.zlog("Start to create Model.")
        # init models
        self.model = layers.BK.new_model()
        # self.nodes = []
        # general computation process: according to the values in the caches
        self.names_bv = set()       # beam variant (need reshuffle within batches)
        self.names_bi = set()       # beam invariant (only controlling the size for each instance will be fine)
        self.names_ig = set()       # ignored (not used in the next steps)
        # properties
        self.props = {}

    @staticmethod
    def new_graph():
        layers.BK.new_graph()

    def get_pc(self):
        # return the real model, only used for Trainer
        return self.model

    def refresh(self, training):
        # should be called after a new graph and before building the graph
        # default: ingraph=True, update=True
        # def _gd(drop):  # get dropout
        #     return drop if training else 0.
        # # disable drop when training; todo(warn) specific names
        # for k in argv:
        #     if k[1:].startwith("drop"):
        #         argv[k] = _gd(argv[k])
        # for n in self.nodes:
        #     n.refresh(argv)
        raise NotImplementedError("Should specify this in specific models!")

    # save and load #
    def load(self, fname):
        self.model = layers.BK.load_model(fname, self.model)
        utils.zlog("Read Model from %s." % fname, func="io")
        try:
            pname = fname+Model.PROP_SUFFIX
            with utils.zopen(pname, 'r') as fd:
                self.props = json.load(fd)
            utils.zlog("Also Read Model-prop from %s." % pname, func="io")
        except:
            # self.props = {}
            pass

    def save(self, fname):
        layers.BK.save_model(fname, self.model)
        utils.zlog("Save Model to %s." % fname, func="io")
        if len(self.props) > 0:
            pname = fname+Model.PROP_SUFFIX
            with utils.zopen(pname, 'w') as fd:
                json.dump(self.props, fd)
            utils.zlog("Also save Model-prop to %s." % pname, func="io")

    def get_prop(self, k):
        if k in self.props:
            return self.props[k]
        else:
            return None

    def set_prop(self, k, v):
        # should be called less often
        v0 = None
        if k in self.props:
            v0 = self.props.pop(k)
        if v is not None:
            self.props[k] = v
        utils.zlog("Change prop %s from %s to %s." % (k, v0, v))

    # main routines #
    def start(self, **kwargs):
        raise NotImplementedError("Should specify this in specific models!")

    def step(self, **kwargs):
        raise NotImplementedError("Should specify this in specific models!")
