# some useful functions
import sys, gzip, platform, subprocess, os, time
import numpy as np
from collections import Iterable

# Part 0: loggings
def zopen(filename, mode='r', encoding="utf-8"):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode, encoding=encoding)
    else:
        return open(filename, mode, encoding=encoding)

def zlog(s, func="plain", flush=True):
    Logger._instance._log(str(s), func, flush)

class Logger(object):
    _instance = None
    _logger_heads = {
        "plain":"-- ", "details":">> ", "time":"## ", "io":"== ", "info":"** ", "score":"%% ",
        "warn":"!! ", "fatal":"KI ", "debug":"DE ", "none":"**INVALID-CODE**"
    }
    @staticmethod
    def _get_ch(func):  # get code & head
        if func not in Logger._logger_heads:
            func = "none"
        return func, Logger._logger_heads[func]
    MAGIC_CODE = "sth_magic_that_cannot_be_conflicted"

    @staticmethod
    def init(files):
        s = "%s-%s.log" % (platform.uname().node, '-'.join(time.ctime().split()[-4:]))
        files = [f if f!=Logger.MAGIC_CODE else s for f in files]
        ff = dict((f, True) for f in files)
        lf = dict((l, True) for l in Logger._logger_heads)
        Logger._instance = Logger(ff, lf)
        zlog("START!!", func="plain")

    # =====
    def __init__(self, file_filters, func_filters):
        self.file_filters = file_filters
        self.func_filters = func_filters
        self.fds = {}
        # the managing of open files (except outside handlers like stdio) is by this one
        for f in self.file_filters:
            if isinstance(f, str):
                self.fds[f] = zopen(f, mode="w")
            else:
                self.fds[f] = f

    def __del__(self):
        for f in self.file_filters:
            if isinstance(f, str):
                self.fds[f].close()

    def _log(self, s, func, flush):
        func, head = Logger._get_ch(func)
        if self.func_filters[func]:
            ss = head + s
            for f in self.fds:
                if self.file_filters[f]:
                    print(ss, file=self.fds[f], flush=flush)

    # todo (register or filter files & codes)

# Part 1: checkers
# Checkings: also including some shortcuts for convenience

def zfatal(ss=""):
    zlog(ss, func="fatal")
    raise RuntimeError()

# general
def zcheck(ff, ss, func="fatal", _forced=False):
    if Checker.enabled() or _forced:
        Checker._instance._check(ff, ss, func, _forced)

# -- len(a) == len(b)
def zcheck_matched_length(a, b, func="fatal", _forced=False):
    if Checker.enabled() or _forced:
        la, lb = len(a), len(b)
        Checker._instance._check(la==lb, "Failed matched_length checking, %s / %s for %s / %s" % (la, lb, type(a), type(b)), func, _forced)

# x in [a, b)
def zcheck_range(x, a=None, b=None, func="fatal", _forced=False):
    if Checker.enabled() or _forced:
        flag = True
        if a is not None and x < a:
            flag = False
        elif b is not None and x >= b:
            flag = False
        Checker._instance._check(flag, "Failed range checking, %s not in [%s, %s)" % (x, a, b), func, _forced)

# type
def zcheck_type(x, t, func="fatal", _forced=False):
    if Checker.enabled() or _forced:
        Checker._instance._check(isinstance(x, t), "Failed in type checking, %s is not %s" % (x, t), func, _forced)

# x in y
def zcheck_in(x, y, func="fatal", _forced=False):
    if Checker.enabled() or _forced:
        Checker._instance._check(x in y, "Failed in checking, %s not in %s" % (x, type(y)), func, _forced)

# function
def zcheck_ff(x, ff, finfo, func="fatal", _forced=False):
    if Checker.enabled() or _forced:
        Checker._instance._check(ff(x), "Failed function checking, %s by %s (%s)" % (x, ff, finfo), func, _forced)

# function_iter
def zcheck_ff_iter(ii, ff, finfo, func="fatal", _forced=False):
    if Checker.enabled() or _forced:
        for i, one in enumerate(ii):
            Checker._instance._check(ff(one), "Failed function-iter checking, %s-th element %s by %s (%s)" % (i, one, ff, finfo), func, _forced)

# should be used when debugging or only fatal ones, comment out if real usage
class Checker(object):
    _instance = None
    _checker_filters = {"warn": True, "fatal": True}
    _checker_handlers = {"warn": (lambda: 0), "fatal": (lambda: zfatal())}
    _checker_enabled = True

    @staticmethod
    def enabled():
        # todo(warn) a better way to disable is to comment out
        return Checker._checker_enabled

    @staticmethod
    def init(enabled):
        Checker._checker_enabled = enabled
        Checker._instance = Checker(Checker._checker_filters, Checker._checker_handlers)

    # =====
    def __init__(self, func_filters, func_handlers):
        self.func_filters = func_filters
        self.func_handlers = func_handlers

    def _check(self, form, ss, func, forced):
        if forced or self._checker_filters[func]:
            if not form:
                zlog(ss, func=func)
                self.func_handlers[func]()

    # todo (manage filters and recordings)

# Part 2: info
def get_statm():
    with zopen("/proc/self/statm") as f:
        rss = (f.read().split())        # strange!! readline-nope, read-ok
        mem0 = str(int(rss[1])*4//1024) + "MiB"
    try:
        p = subprocess.Popen("nvidia-smi | grep -E '%s.*MiB'" % os.getpid(), shell=True, stdout=subprocess.PIPE)
        line = p.stdout.readlines()
        mem1 = line[-1].split()[-2]
    except:
        mem1 = "0MiB"
    return mem0, mem1

# keep times and other stuffs
class Task(object):
    _accu_info = {}  # accumulated info

    @staticmethod
    def get_accu(x=None):
        if x is None:
            return Task._accu_info
        else:
            return Task._accu_info[x]

    def __init__(self, tag, accumulated):
        self.tag = tag
        self.accumulated = accumulated

    def init_state(self):
        raise NotImplementedError()

    def begin(self):
        raise NotImplementedError()

    def end(self, s=None):
        raise NotImplementedError()

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.accumulated:
            if self.tag not in Task._accu_info:
                Task._accu_info[self.tag] = self.init_state()
            Task._accu_info[self.tag] = self.end(Task._accu_info[self.tag])
        else:
            self.end()

class Timer(Task):
    START = 0.

    @staticmethod
    def init():
        Timer.START = time.time()

    @staticmethod
    def systime():
        return time.time()-Timer.START

    def __init__(self, tag, info="", accumulated=False, print_date=False, quiet=False):
        super(Timer, self).__init__(tag, accumulated)
        self.print_date = print_date
        self.quiet = quiet
        self.info = info
        self.accu = 0.   # accumulated time
        self.paused = False
        self.start = Timer.systime()

    def pause(self):
        if not self.paused:
            cur = Timer.systime()
            self.accu += cur - self.start
            self.start = cur
            self.paused = True

    def resume(self):
        if not self.paused:
            zlog("Timer should be paused to be resumed.", func="warn")
        else:
            self.start = Timer.systime()
            self.paused = False

    def get_time(self):
        self.pause()
        self.resume()
        return self.accu

    def init_state(self):
        return 0.

    def begin(self):
        self.start = Timer.systime()
        if not self.quiet:
            cur_date = time.ctime() if self.print_date and not self.quiet else ""
            zlog("Start timer %s: %s at %.3f. (%s)" % (self.tag, self.info, self.start, cur_date), func="time")

    def end(self, s=None):
        self.pause()
        if not self.quiet:
            cur_date = time.ctime() if self.print_date and not self.quiet else ""
            zlog("End timer %s: %s at %.3f, the period is %.3f seconds. (%s)" % (self.tag, self.info, Timer.systime(), self.accu, cur_date), func="time")
        # accumulate
        if s is not None:
            return s+self.accu
        else:
            return None

# Part 3: randomness (all from numpy)
class Random(object):
    _seeds = {}

    @staticmethod
    def get_generator(task):
        if task not in Random._seeds:
            one = 1
            for t in task:
                one = one * ord(t) % (2**30)
            one += 1
            Random._seeds[task] = np.random.RandomState(one)
        return Random._seeds[task]

    @staticmethod
    def init():
        np.random.seed(12345)

    @staticmethod
    def _function(task, type, *argv):
        rg = Random.get_generator(type)
        return getattr(rg, task)(*argv)

    @staticmethod
    def shuffle(xs, type):
        Random._function("shuffle", type, xs)

    @staticmethod
    def binomial(n, p, size, type):
        return Random._function("binomial", type, n, p, size)

    @staticmethod
    def ortho_weight(ndim, type):
        W = Random.randn_clip((ndim, ndim), type)
        u, s, v = np.linalg.svd(W)
        return u.astype(np.float)

    @staticmethod
    def rand(dims, type):
        return Random._function("rand", type, *dims)

    @staticmethod
    def randn_clip(dims, type):
        w = Random._function("randn", type, *dims)
        w.clip(-2, 2)   # clip [-2*si, 2*si]
        return w

    @staticmethod
    def multinomial_select(probs, type):
        # once selection
        x = Random._function("multinomial", type, 1, probs, 1)
        return int(np.argmax(x[0]))

# convenience for basic classes like lists and dicts
class Helper(object):
    @staticmethod
    def combine_dicts(*args, func="fatal", relax=set()):
        # combine {}s, but warn/fatal when finding repeated keys
        x = {}
        for one in args:
            for k in one:
                if k in x and k not in relax:
                    zcheck(False, "Repeated key %s: replacing %s with %s?" % (k, x[k], one[k]), func, _forced=True)
                x[k] = one[k]
        return x

    @staticmethod
    def stream_on_file(fd, tok=(lambda x: x.strip().split())):
        for line in fd:
            for w in tok(line):
                yield w

    @staticmethod
    def stream_rec(obj):
        if isinstance(obj, Iterable):
            for x in obj:
                for y in Helper.stream_rec(x):
                    yield y
        else:
            yield obj

    @staticmethod
    def flatten(obj):
        return list(Helper.stream_rec(obj))

    @staticmethod
    def repeat_list(l, time):
        ret = []
        for one in l:
            for i in range(time):
                ret.append(one)
        return ret

    @staticmethod
    def shrink_list(l, time):
        ret = []
        zcheck(len(l)%time==0, "Unlegal shrink")
        for i in range(len(l)//time):
            ret.append(l[i*time])
        return ret

    @staticmethod
    def add_inplace_list(a, b, mu=1):
        # a += b
        zcheck_matched_length(a, b)
        for i in range(len(a)):
            a[i] += mu*b[i]

    @staticmethod
    def system(cmd, popen=False, print=False, check=False):
        if print:
            zlog("Executing cmd: %s" % cmd)
        if popen:
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            n = p.wait()
            output = p.stdout.read()
            if print:
                zlog("Output is: %s" % output)
        else:
            n = os.system(cmd)
            output = None
        if check:
            assert n==0
        return output

class RangeStater(object):
    def __init__(self, a, b, pieces):
        # [a, b) / pieces
        self.a = a
        self.b = b
        self.pieces = pieces
        self.num = 0
        self.counts = [0] * pieces
        self.interval = (b-a)/pieces
        # end points
        self.counts_a = 0
        self.counts_b = 0

    def add(self, one):
        idx = int((one-self.a)/self.interval)
        idx = max(0, idx)
        idx = min(self.pieces-1, idx)
        self.counts[idx] += 1
        self.num += 1
        if one <= self.a:
            self.counts_a += 1
        elif one >= self.b:
            self.counts_b += 1

    def descr(self):
        divisor = self.num
        if divisor == 0:
            divisor = 1
        return "/".join(["%s(%.3f)"%(x,x/divisor,) for x in self.counts]) + \
               "|ends:(%s(%.3f),%s(%.3f))" % (self.counts_a, self.counts_a/divisor, self.counts_b, self.counts_b/divisor)

# constants
class Constants(object):
    MAX_V = 12345678
    MIN_V = -12345678

# Calling once at start, init them all
def init(extra_file=Logger.MAGIC_CODE):
    flist = [sys.stderr]
    if len(extra_file) > 0:
        flist.append(extra_file)
    Logger.init(flist)
    Checker.init(True)
    Timer.init()
    Random.init()
    # init_print
    zlog("*cmd: %s" % ' '.join(sys.argv))
    zlog("*platform: %s" % ' '.join(platform.uname()))

# ========================================================== #
# outside perspective: init, zlog, zcheck, Random, Timer
