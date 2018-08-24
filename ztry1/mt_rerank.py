# re-ranking with k-best list or possibly with gold ones
# also some kind of analysis from here
from . import mt_args, mt_eval, mt_mt
from zl.data import Vocab, get_arranger_simple
from zl import utils
from .mt_rerank_analysis import BleuCalculator as Analyzer
from .mt_misc import mt_decode

def main():
    # init
    opts = mt_args.init("rerank")
    # special readings from args for re-ranking mode
    # only accept spaced (multi-mode) nbest files for target & non-multi for golds
    # 1. data (only accepting nbest files)
    source_dict, target_dict = Vocab.read(opts["dicts"][0]), Vocab.read(opts["dicts"][1])
    dicts = [source_dict] + [target_dict for _ in opts["test"][1:]]
    test_iter = get_arranger_simple(opts["test"], dicts, multis=[False]+[True for _ in opts["test"][1:]], batch_size=opts["test_batch_size"])
    gold_iter = get_arranger_simple(opts["gold"], [target_dict for _ in opts["gold"]], multis=False, batch_size=opts["test_batch_size"])
    utils.zcheck_matched_length(test_iter, gold_iter)
    # 2. model
    mm = []
    try:
        for mn in opts["models"]:
            x = mt_mt.s2sModel(opts, source_dict, target_dict, None)     # rebuild from opts, thus use the same opts when testing
            try:
                x.load(mn)
            except:
                utils.zlog("Load model error %s!" % mn, func="warn")
            mm.append(x)
    except:
        pass
    # 3. analysis
    if len(mm) == 0:
        utils.zlog("No models specified, only analysing!", func="warn")
        num_test = len(opts["test"])-1
        golds = []
        srcs = []
        preds = [[] for _ in range(num_test)]
        for one in gold_iter.arrange_batches():
            golds += one
        for one in test_iter.arrange_batches():
            for zz in one:
                zzs = zz.extract()
                srcs.append(zzs[0])
                for i in range(num_test):
                    preds[i].append(zzs[i+1])
        Analyzer.analyse(srcs, golds, preds, kbests=opts["rr_analysis_kbests"])
    # 4. rerank
    else:
        utils.zlog("=== Start to rerank ===", func="info")
        with utils.Timer(tag="Reranking", print_date=True):
            mt_decode(None, test_iter, mm, target_dict, opts, opts["output"], gold_iter=gold_iter)
        utils.zlog("=== End reranking, write to %s ===" % opts["output"], func="info")
        mt_eval.evaluate(opts["output"], opts["gold"][0], opts["eval_metric"])

if __name__ == '__main__':
    main()
