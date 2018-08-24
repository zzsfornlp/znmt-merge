import sys
from . import mt_args, mt_search, mt_eval
from zl import utils
from zl.data import Vocab, get_arranger, TextInstance
from .mt_mt import s2sModel
from .mt_misc import mt_decode
from .mt_outputter import Outputter

def main():
    # init
    opts = mt_args.init("test")
    looping = opts["loop"]
    # 1. data
    source_dict, target_dict = Vocab.read(opts["dicts"][0]), Vocab.read(opts["dicts"][1])
    # -- here usually no need for test[1], but for convenience ...
    if not looping:
        dicts = [source_dict] + [target_dict for _ in opts["test"][1:]]
        test_iter = get_arranger(opts["test"], dicts, multis=False, shuffling_corpus=False, shuflling_buckets=False, sort_prior=[0], batch_size=opts["test_batch_size"], maxibatch_size=-1, max_len=utils.Constants.MAX_V, min_len=0, one_len=opts["max_len"]+1, shuffling0=False)
    # 2. model
    mm = []
    for mn in opts["models"]:
        x = s2sModel(opts, source_dict, target_dict, None)     # rebuild from opts, thus use the same opts when testing
        x.load(mn)
        mm.append(x)
    if len(mm) == 0:
        utils.zlog("No models specified, must be testing mode?", func="warn")
        mm.append(s2sModel(opts, source_dict, target_dict, None))      # no loading, only for testing
    # 3. decode
    if not looping:
        utils.zlog("=== Start to decode ===", func="info")
        with utils.Timer(tag="Decoding", print_date=True):
            mt_decode(opts["decode_way"], test_iter, mm, target_dict, opts, opts["output"])
        utils.zlog("=== End decoding, write to %s ===" % opts["output"], func="info")
        # todo(warn) forward-compatible evaluation
        if len(opts["test"]) > 1:
            gold = opts["test"][1]
        else:
            gold = opts["gold"][0]
        mt_eval.evaluate(opts["output"], gold, opts["eval_metric"])
    else:
        ot = Outputter(opts)
        while True:
            utils.zlog("Enter the src to translate:")
            line = sys.stdin.readline()
            if len(line)==0:
                break
            # prepare one
            one_words = line.strip().split()
            one_idxes = Vocab.w2i(source_dict, one_words, add_eos=True, use_factor=False)
            one_inst = TextInstance([one_words], [one_idxes])
            rs = mt_decode(opts["decode_way"], [one_inst], mm, target_dict, opts, opts["output"])
            utils.zlog(ot.format(rs[0], target_dict, False, False))

if __name__ == '__main__':
    main()
