from . import mt_args
from zl import utils
from zl.data import Vocab, get_arranger
from .mt_mt import s2sModel
from .mt_misc import MTTrainer
from .mt_length import LinearGaussain
import os

def main():
    # init
    opts = mt_args.init("train")
    # start to train
    # 1. obtain dictionaries
    source_corpus, target_corpus = opts["train"]
    source_dict, target_dict = None, None
    if not opts["rebuild_dicts"]:
        try:
            source_dict, target_dict = Vocab.read(opts["dicts"][0]), Vocab.read(opts["dicts"][1])
        except:
            utils.zlog("Read dictionaries fail %s, rebuild them." % (opts["dicts"],), func="warn")
    if source_dict is None or target_dict is None:
        # rebuild the dictionaries from corpus
        source_dict = Vocab(fname=source_corpus, rthres=opts["dicts_rthres"], fthres=opts["dicts_fthres"])
        target_dict = Vocab(fname=target_corpus, rthres=opts["dicts_rthres"], fthres=opts["dicts_fthres"])
        # save dictionaries
        try:
            source_dict.write(opts["dicts"][0])
            target_dict.write(opts["dicts"][1])
        except:
            utils.zlog("Write dictionaries fail: %s, skip this step." % opts["dicts_final"], func="warn")
    # 2. corpus iterator
    shuffling0 = opts["shuffle_training_data_onceatstart"]
    sort_prior = {"src":[0], "trg":[1], "src-trg":[0,1], "trg-src":[1,0]}[opts["training_sort_type"]]
    train_iter = get_arranger(opts["train"], [source_dict, target_dict], multis=False, shuffling_corpus=opts["shuffle_training_data"], shuflling_buckets=opts["shuffle_training_data"], sort_prior=sort_prior, batch_size=opts["batch_size"], maxibatch_size=20, max_len=opts["max_len"]+1, min_len=2, one_len=opts["max_len"]+1, shuffling0=shuffling0)
    dev_iter = get_arranger(opts["dev"], [source_dict, target_dict], multis=False, shuffling_corpus=False, shuflling_buckets=False, sort_prior=[0], batch_size=opts["valid_batch_size"], maxibatch_size=-1, max_len=utils.Constants.MAX_V, min_len=0, one_len=opts["max_len"]+1, shuffling0=False)
    # 3. about model & trainer
    # <special one> fit a gaussian first
    length_info = LinearGaussain.fit_once(opts["train"])   # todo: train or dev?
    mm = s2sModel(opts, source_dict, target_dict, length_info)
    tt = MTTrainer(opts, mm)  # trainer + training_progress
    if opts["reload"] and os.path.exists(opts["reload_model_name"]):
        tt.load(opts["reload_model_name"], opts["reload_training_progress"])
    # 4. training
    tt.train(train_iter, dev_iter)
    utils.zlog("=== Training ok!! ===", func="info")

if __name__ == '__main__':
    main()
