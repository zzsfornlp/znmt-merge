from . import utils
from .layers import BK
import json

class TrainingProgress(object):
    # Object used to store, serialize and deserialize pure python variables that change during training and should be preserved in order to properly restart the training process
    # for score, the larger the better
    def load_from_json(self, file_name):
        with utils.zopen(file_name, 'r') as fd:
            self.__dict__.update(json.load(fd))

    def save_to_json(self, file_name):
        with utils.zopen(file_name, 'w') as fd:
            json.dump(self.__dict__, fd, indent=2)

    def __init__(self, patience, anneal_restarts):
        self.bad_counter = 0
        self.bad_points = []
        self.anneal_restarts_done = 0
        self.anneal_restarts_points = []
        self.uidx = 0                   # update
        self.eidx = 0                   # epoch
        self.estop = False
        self.hist_points = []
        self.hist_scores = []            # the bigger the better
        self.hist_trains = []
        self.best_score = None
        self.best_point = None
        # states
        self.patience = patience
        self.anneal_restarts_cap = anneal_restarts
        #
        self.hists = []
        self.sortings = []

    @property
    def num_anneals(self):
        return self.anneal_restarts_done

    def record(self, ss, score, trains):
        def _getv(z):
            return utils.Constants.MIN_V if z is None else z[0]
        # record one score, return (if-best, if-anneal)
        if_best = if_anneal = False
        self.hist_points.append(ss)
        self.hist_scores.append(score)
        self.hist_trains.append(trains)
        if _getv(score) > _getv(self.best_score):
            self.bad_counter = 0
            self.best_score = score
            self.best_point = ss
            if_best = True
        else:
            self.bad_counter += 1
            self.bad_points.append([ss, self.best_score, self.best_point])      # bad_point, .best-at-that-time
            utils.zlog("Bad++, now bad/anneal is %s/%s." % (self.bad_counter, self.anneal_restarts_done), func="info")
            if self.bad_counter >= self.patience:
                self.bad_counter = 0
                self.anneal_restarts_points.append([ss, self.best_score, self.best_point])
                if self.anneal_restarts_done < self.anneal_restarts_cap:
                    self.anneal_restarts_done += 1
                    utils.zlog("Anneal plus one, now %s." % (self.anneal_restarts_done,), func="info")
                    if_anneal = True
                else:
                    utils.zlog("Sorry, Early Update !!", func="warn")
                    self.estop = True
        #
        self.hists = [(_1, _getv(_2)) for _1,_2 in zip(self.hist_points, self.hist_scores)]
        self.sortings = sorted(self.hists, key=lambda x: x[-1], reverse=True)[:10]
        return if_best, if_anneal

    def report(self):
        def _getv(z):
            return utils.Constants.MIN_V if z is None else z[0]
        for k in sorted(self.__dict__):
            utils.zlog("Training progress results: %s = %s." % (k, self.__dict__[k]), func="score")
        # utils.zlog("Ranking top10 is: %s" % self.sortings)

    def link_bests(self, basename):
        utils.Helper.system("rm ztop_model*")
        for i, pair in enumerate(self.sortings):
            utils.Helper.system("ln -s %s%s ztop_model%s" % (basename, pair[0], i), print=True)

# class ValidScore(object):
#     pass

class Trainer(object):
    TP_TAIL = ".progress.json"
    TR_TAIL = ".trainer.shadow"
    ANNEAL_DECAY = 0.5
    CURR_PREFIX = "zcurr."
    BEST_PREFIX = "zbest."

    def __init__(self, opts, model):
        # Used options: lrate, moment, trainer_type, clip_c, max_epochs, max_updates, overwrite
        # --: anneal_renew_trainer, anneal_reload_best, anneal_restarts, patience
        self.opts = opts
        self._tp = TrainingProgress(opts["patience"], opts["anneal_restarts"])
        self._mm = model
        self.trainer = None
        self._set_trainer(True)

    def _set_trainer(self, renew):
        cur_lr = self.opts["lrate"] * (Trainer.ANNEAL_DECAY ** self._tp.num_anneals)
        if self.trainer is None:
            self.trainer = BK.Trainer(self._mm.get_pc(), self.opts["trainer_type"], self.opts["lrate"], self.opts["moment"])
        elif renew:
            self.trainer.restart()
        self.trainer.set_lrate(cur_lr)
        self.trainer.set_clip(self.opts["clip_c"])
        utils.zlog("Set trainer %s with lr %s." % (self.opts["trainer_type"], cur_lr), func="info")

    # load and save
    def load(self, basename, load_process):
        # load model
        self._mm.load(basename)
        utils.zlog("Reload model from %s." % basename, func="io")
        # load progress
        if load_process:
            tp_name = basename + Trainer.TP_TAIL
            tr_name = basename + Trainer.TR_TAIL
            utils.zlog("Reload trainer from %s and %s." % (tp_name, tr_name), func="io")
            self._tp.load_from_json(tp_name)
            if self._finished():
                utils.zfatal('Training is already complete. Disable reloading of training progress (--no_reload_training_progress)'
                            'or remove or modify progress file (%s) to train anyway.')
            # renew & set learning rate
            self._set_trainer(True)
            # TODO, possibly load-shadows

    def save(self, basename):
        # save model
        self._mm.save(basename)
        # save progress
        tp_name = basename + Trainer.TP_TAIL
        tr_name = basename + Trainer.TR_TAIL
        utils.zlog("Save trainer to %s and %s." % (tp_name, tr_name), func="io")
        self._tp.save_to_json(tp_name)
        self.trainer.save_shadow(tr_name)

    # helpers
    def _finished(self):
        return self._tp.estop or self._tp.eidx >= self.opts["max_epochs"] \
                or self._tp.uidx >= self.opts["max_updates"]

    def _update_before(self):
        self._mm.update_schedule(self._tp.uidx)

    def _update_after(self):
        self.trainer.update()
        self._tp.uidx += 1

    # ========
    def _validate_them(self, dev_iter, metrics):
        raise NotImplementedError("Not here are basic trainer!")

    def _get_recorder(self, name):
        raise NotImplementedError("Not here are basic trainer!")

    def _fb_once(self, insts):
        raise NotImplementedError("Not here are basic trainer!")

    # main rountines
    def _validate(self, dev_iter, name=None, training_states=None):
        # validate and log in the stats
        ss = ".e%s-u%s" % (self._tp.eidx, self._tp.uidx) if name is None else name
        with utils.Timer(tag="valid", info="Valid %s" % ss, print_date=True):
            # checkpoint - write current
            self.save(Trainer.CURR_PREFIX+self.opts["model"])
            if not self.opts["overwrite"]:
                self.save(self.opts["model"]+ss)
            # validate
            score = self._validate_them(dev_iter, self.opts["valid_metrics"])
            utils.zlog("Validating %s for %s: score is %s." % (self.opts["valid_metrics"], ss, score), func="score")
            # write best and update stats
            if_best, if_anneal = self._tp.record(ss, score, training_states)
            if if_best:
                self.save(Trainer.BEST_PREFIX+self.opts["model"])
            if if_anneal:
                if self.opts["anneal_reload_best"]:
                    self.load(Trainer.BEST_PREFIX+self.opts["model"], False)   # load model, but not process
                self._set_trainer(self.opts["anneal_renew_trainer"])
        self._tp.report()
        self._tp.link_bests(self.opts["model"])
        utils.zlog("", func="info")     # to see it more clearly

    # main training
    def train(self, train_iter, dev_iter):
        one_recorder = self._get_recorder("CHECK")
        if self.opts["validate0"]:
            self._validate(dev_iter, training_states=one_recorder.state())      # validate once before training
        while not self._finished():     # epochs
            # utils.printing("", func="info")
            with utils.Timer(tag="Train-Iter", info="Iter %s" % self._tp.eidx, print_date=True) as et:
                iter_recorder = self._get_recorder("ITER-%s" % self._tp.eidx)
                for insts in train_iter.arrange_batches():
                    if utils.Random.rand([1], "skipping") < self.opts["rand_skip"]:     # introduce certain randomness
                        continue
                    # training for one batch
                    self._update_before()
                    loss = self._fb_once(insts)
                    self._update_after()
                    one_recorder.record(insts, loss, 1)
                    iter_recorder.record(insts, loss, 1)
                    if self.opts["debug"] and self.opts["verbose"]:
                        mem0, mem1 = utils.get_statm()
                        utils.zlog("[%s/%s] after fb(%s): %s" % (mem0, mem1, len(insts), insts[0].describe(insts)), func="DE")
                    if self.opts["verbose"] and self._tp.uidx % self.opts["report_freq"] == 0:
                        one_recorder.report("Training process: ")
                    # time to validate and save best model ??
                    if self.opts["validate_freq"] and self._tp.uidx % self.opts["valid_freq"] == 0:    # update when _update
                        one_recorder.report()
                        self._validate(dev_iter, training_states=one_recorder.state())
                        one_recorder.reset()
                        if self._finished():
                            break
                iter_recorder.report()
                if self.opts["validate_epoch"]:
                    # here, also record one_recorder, might not be accurate
                    one_recorder.report()
                    self._validate(dev_iter, name=".ev%s" % self._tp.eidx, training_states=one_recorder.state())
                    one_recorder.reset()
                self._tp.eidx += 1
