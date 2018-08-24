import argparse
import zl
import numpy as np

# parse the arguments for main
def init(phase):
    parser = argparse.ArgumentParser()

    data = parser.add_argument_group('data sets; model loading and saving')
    if phase == "train":
        # -- data sets and dictionaries
        data.add_argument('--train', type=str, required=True, metavar='PATH', nargs=2,
                             help="parallel training corpus (source and target)")
        data.add_argument('--dev', type=str, required=True, metavar='PATH', nargs=2,
                             help="parallel dev corpus (source and target)")
        data.add_argument('--dev_output', type=str, default="dev-output.txt",
                             help="output target corpus for dev (if needed)")
        data.add_argument('--dicts', type=str, default=["src.v", "trg.v"], metavar='PATH', nargs="+",
                             help="final dictionaries (source / target), also write dest")
        data.add_argument('--no_rebuild_dicts', action='store_false', dest='rebuild_dicts',
                             help="rebuild dictionaries and write to files")
        data.add_argument('--dicts_rthres', type=int, default=50000, metavar='INT',
                             help="cutting threshold by rank (<= rthres) (default: %(default)s)")
        data.add_argument('--dicts_fthres', type=int, default=1, metavar='INT',
                             help="cutting threshold by freq (>= rfreq) (default: %(default)s)")
        # -- about model -- save and load
        data.add_argument('--model', type=str, default='model', metavar='PATH',
                             help="model file name (default: %(default)s)")
        data.add_argument('--reload', action='store_true',
                             help="load existing model (if '--reload_model_name' points to existing model)")
        data.add_argument('--reload_model_name', type=str, metavar='PATH',
                             help="reload model file name (default: %(default)s)")
        data.add_argument('--no_reload_training_progress', action='store_false',  dest='reload_training_progress',
                             help="don't reload training progress (only used if --reload is enabled)")
        data.add_argument('--no_overwrite', action='store_false', dest='overwrite',
                             help="don't write all models to same file")
    elif phase == "test" or phase == "rerank":
        # data, dictionary, model
        data.add_argument('--test', '-t', type=str, required=True, metavar='PATH', nargs="+",
                             help="parallel testing corpus, maybe multiple targets for reranking (source and target)")
        data.add_argument('--output', '-o', type=str, default='output.txt', metavar='PATH', help="output target corpus")
        data.add_argument('--gold', type=str, metavar='PATH', help="gold target corpus (for eval)", nargs="+")
        data.add_argument('--dicts', '-d', type=str, default=["src.v", "trg.v"], metavar='PATH', nargs="+",
                          help="final dictionaries (source / target)")
        data.add_argument('--models', '-m', type=str, default=["zbest.model"], metavar='PATHs', nargs="*",
                             help="model file names (ensemble if >1)")
        data.add_argument("--loop", action='store_true', help="Entering looping mode, reading from std-inputs.")
        # ----------
        # -- no used options, just for convenience of the train-test scripts
        data.add_argument('--dicts_rthres', type=int, default=50000, metavar='INT', help="NON-USED OPTION")
        data.add_argument('--dicts_fthres', type=int, default=1, metavar='INT', help="NON-USED OPTION")
        data.add_argument('--no_rebuild_dicts', action='store_false', help="NON-USED OPTION")
        data.add_argument('--reload', action='store_true', help="NON-USED OPTION")
        data.add_argument('--reload_model_name', type=str, help="NON-USED OPTION")
        data.add_argument('--no_reload_training_progress', action='store_false', help="NON-USED OPTION")
    else:
        raise NotImplementedError(phase)

    # architecture
    network = parser.add_argument_group('network parameters')
    network.add_argument('--dim_word', type=int, default=512, metavar='INT',
                         help="embedding layer size (default: %(default)s)")
    network.add_argument('--dec_type', type=str, default="nematus", choices=["att", "nematus", "ngram"],
                         help="decoder type (default: %(default)s)")
    network.add_argument('--dec_ngram_n', type=int, default=5, help="Ngram if using ngram-decoder (default: %(default)s)")
    network.add_argument('--att_type', type=str, default="ff", choices=["ff", "biaff", "dummy"],
                         help="attention type (default: %(default)s)")
    network.add_argument('--rnn_type', type=str, default="gru", choices=["gru", "gru2"],
                         help="recurrent node type (default: %(default)s)")
    network.add_argument('--summ_type', type=str, default="ends", choices=["avg", "fend", "bend", "ends"],
                         help="decoder's starting summarizing type (default: %(default)s)")
    network.add_argument('--hidden_rec', type=int, default=1000, metavar='INT',
                         help="recurrent hidden layer (default for dec&&enc) (default: %(default)s)")
    network.add_argument('--hidden_dec', type=int, metavar='INT',
                         help="decoder hidden layer size (default: hidden_rec")
    network.add_argument('--hidden_enc', type=int, metavar='INT',
                         help="encoder hidden layer size <BiRNN thus x2> (default: hidden_rec")
    network.add_argument('--hidden_att', type=int, default=1000, metavar='INT',
                         help="attention hidden layer size (default: %(default)s)")
    network.add_argument('--hidden_out', type=int, default=500, metavar='INT',
                         help="output hidden layer size (default: %(default)s)")
    # network.add_argument('--dim_cov', type=int, default=0, metavar='INT',
    #                      help="dimension for coverage in att (default: %(default)s)")
    network.add_argument('--enc_depth', type=int, default=1, metavar='INT',
                         help="number of encoder layers (default: %(default)s)")
    network.add_argument('--dec_depth', type=int, default=1, metavar='INT',         # only the first is with att
                         help="number of decoder layers (default: %(default)s)")
    network.add_argument('--drop_hidden', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for hidden layers (0: no dropout) (default: %(default)s)")
    network.add_argument('--drop_embedding', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for embeddings (0: no dropout) (default: %(default)s)")
    network.add_argument('--idrop_embedding', type=float, default=0., metavar="FLOAT",
                         help="idrop for words (0: no dropout) (default: %(default)s)")
    network.add_argument('--gdrop_embedding', type=float, default=0., metavar="FLOAT",
                         help="gdrop for words (0: no dropout) (default: %(default)s)")
    network.add_argument('--idrop_rec', type=float, default=0., metavar="FLOAT",
                         help="dropout (idrop) for recurrent nodes (0: no dropout) (default: %(default)s)")
    network.add_argument('--idrop_dec', type=float, metavar="FLOAT",
                         help="dropout (idrop) for decoder (0: no dropout) (default: %(default)s)")
    network.add_argument('--idrop_enc', type=float, metavar="FLOAT",
                         help="dropout (idrop) for encoder (0: no dropout) (default: %(default)s)")
    network.add_argument('--gdrop_rec', type=float, default=0., metavar="FLOAT",
                         help="gdrop for recurrent nodes (0: no dropout) (default: %(default)s)")
    network.add_argument('--gdrop_dec', type=float, metavar="FLOAT",
                         help="gdrop for decoder (0: no dropout) (default: %(default)s)")
    network.add_argument('--gdrop_enc', type=float, metavar="FLOAT",
                         help="gdrop for encoder (0: no dropout) (default: %(default)s)")
    # special option
    network.add_argument('--drop_test', action="store_true", help="Special option for opening dropout for testing.")

    # training progress
    training = parser.add_argument_group('training parameters')
    training.add_argument('--shuffle_training_data_onceatstart', action='store_true', help="first shuffle before training.")
    training.add_argument('--no_shuffle_training_data', action='store_false', dest='shuffle_training_data',
                             help="don't shuffle training data before each epoch")
    network.add_argument('--training_sort_type', type=str, default="trg-src", choices=["src", "trg", "src-trg", "trg-src"],
                         help="training data's sort type (default: %(default)s)")
    training.add_argument('--max_len', type=int, default=50, metavar='INT',
                         help="maximum sequence length (default: %(default)s)")
    training.add_argument('--batch_size', type=int, default=80, metavar='INT',
                         help="minibatch size (default: %(default)s)")
    training.add_argument('--rand_skip', type=float, default=0., metavar='INT',
                         help="randomly skip batches for training (default: %(default)s)")
    training.add_argument('--max_epochs', type=int, default=50, metavar='INT',
                         help="maximum number of epochs (default: %(default)s)")
    training.add_argument('--max_updates', type=int, default=1000000, metavar='INT',
                         help="maximum number of updates (minibatches) (default: %(default)s)")
    # -- trainer
    network.add_argument('--trainer_type', type=str, default="adam", choices=["adam", "sgd", "momentum"],
                         help="trainer type (default: %(default)s)")
    training.add_argument('--clip_c', type=float, default=1, metavar='FLOAT',
                         help="gradient clipping threshold (default: %(default)s)")
    training.add_argument('--lrate', type=float, default=0.0001, metavar='FLOAT',
                         help="learning rate or alpha (default: %(default)s)")
    training.add_argument('--moment', type=float, default=0.8, metavar='FLOAT',
                         help="momentum for mTrainer (default: %(default)s)")

    # -- validate
    validation = parser.add_argument_group('validation parameters')
    validation.add_argument('--valid_freq', type=int, default=10000, metavar='INT',
                         help="validation frequency (default: %(default)s)")
    validation.add_argument('--valid_batch_size', '--valid_batch_width', type=int, default=40, metavar='INT',
                         help="validating minibatch-size (default: %(default)s)")
    validation.add_argument('--patience', type=int, default=10, metavar='INT',
                         help="early stopping patience (default: %(default)s)")
    validation.add_argument('--anneal_restarts', type=int, default=0, metavar='INT',
                         help="when patience runs out, restart training INT times with annealed learning rate (default: %(default)s)")
    validation.add_argument('--anneal_no_renew_trainer', action='store_false',  dest='anneal_renew_trainer',
                         help="don't renew trainer (discard moments or grad info) when anneal")
    validation.add_argument('--anneal_no_reload_best', action='store_false',  dest='anneal_reload_best',
                         help="don't recovery to previous best point (discard some training) when anneal")
    validation.add_argument('--anneal_decay', type=float, default=0.5, metavar='FLOAT',
                         help="learning rate decay on each restart (default: %(default)s)")
    validation.add_argument('--valid_metrics', type=str, default="bleu,ll,len",
                         help="type of metric for validation (separated by ',') (default: %(default)s)")
    validation.add_argument('--validate_epoch', action='store_true', help="validate at the end of each epoch")
    validation.add_argument('--validate0', action='store_true', help="validate at the start")
    validation.add_argument('--no_validate_freq', action='store_false', dest="validate_freq", help="no validating at freq points.")

    # common
    common = parser.add_argument_group('common')
    # -- dynet
    common.add_argument("--dynet-mem", type=str, default="4", dest="dynet-mem")
    common.add_argument("--dynet-devices", type=str, default="CPU", dest="dynet-devices")
    common.add_argument("--dynet-autobatch", type=str, default="0", dest="dynet-autobatch")
    common.add_argument("--dynet-seed", type=str, default="12345", dest="dynet-seed")
    common.add_argument("--dynet-immed", action='store_true', dest="dynet-immed")
    # -- bk init
    common.add_argument("--bk_init_enabled", action='store_true')
    common.add_argument("--bk_init_nl", type=str)
    common.add_argument("--bk_init_l", type=str)
    # -- others
    common.add_argument("--debug", action='store_true')
    common.add_argument("--verbose", "-v", action='store_true')
    common.add_argument("--log", type=str, default=zl.utils.Logger.MAGIC_CODE, help="logger for the process")
    common.add_argument('--report_freq', type=int, default=1000, metavar='INT',
                         help="report frequency (number of instances / only when verbose) (default: %(default)s)")

    # decode (also for BLEU validation, training with other params)
    decode = parser.add_argument_group('decode')
    # decode.add_argument('--decode_type', '--decode_mode', type=str, default="decode", choices=["decode", "decode_gold", "test1", "test2", "loop"],
    #                      help="type/mode of testing (decode, test, loop)")
    decode.add_argument('--decode_way', type=str, default="beam", choices=["greedy", "beam", "sample", "branch"],
                         help="decoding method (default: %(default)s)")
    decode.add_argument('--beam_size', '-k', type=int, default=10, help="Beam size (default: %(default)s))")
    decode.add_argument('--decode_len', type=int, default=80, metavar='INT',
                         help="maximum decoding sequence length (default: %(default)s)")
    decode.add_argument('--decode_ratio', type=float, default=5.,
                         help="maximum decoding sequence length ratio compared with src (default: %(default)s)")
    decode.add_argument('--eval_metric', type=str, default="bleu", choices=["bleu", "ibleu", "nist"],
                         help="type of metric for evaluation (default: %(default)s)")
    decode.add_argument('--test_batch_size', type=int, default=8, metavar='INT',
                         help="testing minibatch-size(default: %(default)s)")

    # extra: for advanced training
    training2 = parser.add_argument_group('training parameters section2')
    # scale original loss function for training
    training2.add_argument('--train_scale', type=float, default=0.0, metavar="ALPHA",
                         help="(train2) Scale scores by sentence length (exponentiate lengths by ALPHA, neg means nope)")
    training2.add_argument('--train_scale_way', type=str, default="none", choices=["none", "norm", "google"],
                         help="(train2) how to norm length with score scales (default: %(default)s)")
    # length fitting for training
    # todo(warn): these options are also used for decoding process
    training2.add_argument('--train_len_uidx', type=int, default=10000000,
                         help="start to fit len after this updates (default: %(default)s)")
    training2.add_argument('--train_len_lambda', type=float, default=1.0, help="lambda for length loss (default: %(default)s)")
    training2.add_argument('--train_len_xadd', action='store_true', help="adding xsrc for length fitting")
    training2.add_argument('--train_len_xback', action='store_true', help="backprop of xsrc for length fitting")
    # -- training methods and output modeling
    training2.add_argument('--no_model_softmax', action='store_true', help="No adding softmax for vocab output (direct score)")
    training2.add_argument('--train_r2l', action='store_true', help="training right to left model")
    training2.add_argument('--train_mode', type=str, default="std2", choices=["std2", "beam", "branch"], help="which training process?")
    training2.add_argument('--train_local_loss', type=str, default="mle", choices=["mle", "mlev", "hinge_max", "hinge_avg", "hinge_avg0", "hinge_sum"], help="Training objective.")
    training2.add_argument('--train_margin', type=float, default=0., help="The margin for margin-training.")
    # # # # #
    # -- start training with non-gold seq (reusing opts from other parts)
    # === fb_beam
    training2.add_argument('--t2_search_ratio', type=float, default=1.28, help="Max search steps ratio according to reference.")
    training2.add_argument('--t2_gold_run', action='store_true', help="First running a gold sequence.")
    training2.add_argument('--t2_beam_size', type=int, default=1, help="Beam size for beam training2.")
    training2.add_argument('--t2_impl_bsize', type=int, default=40, help="Impl bsize for fb_beam.")
    training2.add_argument('--t2_beam_nodrop', action='store_true', help="First beam search with no drops and rerun gold & beam with dropouts.")
    #
    training2.add_argument('--t2_local_expand', type=int, default=100, help="Most expansions for each state.")
    training2.add_argument('--t2_local_diff', type=float, default=100., help="Local pruning diff/thres (1/e**D if transferring to prob.)")
    training2.add_argument('--t2_global_expand', type=int, default=100, help="How many states could survive in one global-beam.")
    training2.add_argument('--t2_global_diff', type=float, default=100., help="Global pruning normalized diff/thres (1/e**D if transferring to prob.)")
    training2.add_argument('--t2_bngram_n', type=int, default=5, help="Nth tailing ngram sig for pruning.")
    training2.add_argument('--t2_bngram_range', type=int, default=0, help="Number of the range of history for tngram, 0 for none.")
    #
    # == synchronization (med or nga or [todo]both?)
    training2.add_argument('--t2_sync_med', action='store_true', help="Fb-beam with med sync.")
    training2.add_argument('--t2_med_nosub', action='store_true', help="No sub when calculating med.")
    training2.add_argument('--t2_med_range', type=int, default=100, help="Range of med, default might be enough, n for 2n-1")
    training2.add_argument('--t2_sync_nga', action='store_true', help="Fb-beam with nga sync.")
    training2.add_argument('--t2_nga_n', type=int, default=5, help="Nth tailing ngram sig for matching gold.")
    training2.add_argument('--t2_nga_range', type=int, default=0, help="Number of the range of history for matching gold, 0 for none, n for 2n-1.")
    #
    # == gold interference (suggest using up=end if setting GI)
    # -> mode: laso=laso, nga=nga-best, none=means single updating points
    training2.add_argument('--t2_gi_mode', type=str, default="none", choices=["none", "laso", "ngab"], help="How gold will influence the learning")
    #
    # == loss function and updating
    # -> the final loss: perceptron; local-prob-with-err-states
    training2.add_argument('--t2_beam_loss', type=str, default="err", choices=["per", "err"], help="what is the loss for fb_beam?")
    training2.add_argument('--t2_bad_lambda', type=float, default=1.0, help="Scaling factor for bad-seq loss.")
    training2.add_argument('--t2_bad_maxlen', type=int, default=100, help="Max seq-length for bad-seq loss.")
    # >> especailly for err mode
    training2.add_argument('--t2_err_gold_mode', type=str, default="no", choices=["no","gold","based"], help="How about err-s's corresponding golds.")
    training2.add_argument('--t2_err_gold_lambda', type=float, default=1., help="Lambda for traditional gold loss.")
    training2.add_argument('--t2_err_pred_lambda', type=float, default=1., help="Lambda for pred's loss.")
    training2.add_argument('--t2_err_match_nope', action='store_true', help="No loss for the matched pred seg.")
    training2.add_argument('--t2_err_match_addfirst', action='store_true', help="Add the first token for matched seg.")
    training2.add_argument('--t2_err_match_addeos', action='store_true', dest="t2_err_match_addeos", help="Add eos for matched seg.")
    training2.add_argument('--t2_err_cor_nofirst', action='store_true', help="Do not include first token of correction for bad seq.")
    # >> thresh
    training2.add_argument('--t2_err_mcov_thresh', type=float, default=0., help="Original loss if matched cover less than this thresh.")
    training2.add_argument('--t2_err_pright_thresh', type=float, default=1.01, help="Consider pred as gold if matched >= this one.")
    training2.add_argument('--t2_err_thresh_bleu', action='store_true', help="Using bleu rather than matched ratio as thresh criterion.")
    # >>> special control for threshing
    training2.add_argument('--t2_err_seg_minlen', type=int, default=1, help="Ignore & combine segs whose length is less than this thresh.")
    training2.add_argument('--t2_err_seg_freq_token', type=int, default=0, help="Regard as freq token if <= this thresh.")
    training2.add_argument('--t2_err_seg_extend_range', type=int, default=0, help="Checking range for extends for a matched seg.")
    #
    training2.add_argument('--t2_err_debug_print', action='store_true')
    # -> updating points: early-update(first-gi-point:FG); at-end(all-gi-points or end points:AG)
    training2.add_argument('--t2_beam_up', type=str, default="ag", choices=["ag", "fg"], help="The updating points")
    # -> how to compare the state scores for attached points
    training2.add_argument('--t2_compare_at', type=str, default="norm", choices=["norm", "none"], help="Normalizing methods for fb_beam")

    # extra: for advanced decoding
    decode2 = parser.add_argument_group('decoding parameters section2')
    # -- general
    decode2.add_argument('--no_output_kbest', action='store_false', help="Output special files with all outputs.", dest="decode_output_kbest")
    decode2.add_argument('--decode_write_once', action='store_true', help="Output the results only once at the end (need larger cpu mem to store the results.)")
    decode2.add_argument('--decode_dump_hiddens', action='store_true', help="Dump hiddens for all states.")
    decode2.add_argument('--decode_dump_sg', action='store_true', help="Dump search graphs.")
    decode2.add_argument('--decode_replace_unk', action='store_true', help="Copy max-attention src for UNK.")
    decode2.add_argument('--decode_latnbest', action='store_true', help="Re-generate n-best from lattice.")
    decode2.add_argument('--decode_latnbest_nalpha', type=float, default=0.0, help="Length normalizer for lattice n-best.")
    decode2.add_argument('--decode_latnbest_lreward', type=float, default=0.0, help="Length reward for lattice n-best.")
    decode2.add_argument('--decode_latnbest_rtimes', type=int, default=1, help="Maximum repeating times of link in the stack.")
    decode2.add_argument('--decode_output_r2l', action='store_true', help="r2l model")
    # -- para_extractor
    decode2.add_argument('--decode_extract_paraf', action='store_true', help="Extract paraf as a by-product when decoding.")
    # -> re-using t2_*
    # decode2.add_argument('--decode_paraf_nosub', action='store_true', help="Equiv to t2_med_nosub.")
    # decode2.add_argument('--decode_paraf_matched_minlen', type=int, default=1,  help="Equiv to t2_err_seg_minlen.")
    # -- norm
    # todo(warn): have to be cautious about parameters, some model specification is also needed to construct model for decoding
    # todo(warn): only using the first model if using gaussian
    decode2.add_argument('--normalize_way', type=str, default="none", choices=["none", "norm", "google", "add", "gaussian", "xgaussian"],
                         help="how to norm length (default: %(default)s)")
    decode2.add_argument('--normalize_alpha', '-n', type=float, default=0.0, metavar="ALPHA",
                         help="Normalize scores by sentence length or lambda for gaussian.")
    decode2.add_argument('--penalize_eos', type=float, default=0.0, help="Directly penalizing scores when decoding of EOS & '.'.")
    # -- pruning
    # --- length (the default ones should be enough)
    decode2.add_argument('--pr_len_khigh', type=float, default=10., metavar="K_HIGH",
                         help="Another max-len upper limit: (mu+k*si)")
    decode2.add_argument('--pr_len_klow', type=float, default=10., metavar="K_LOW",
                         help="Another max-len lower limit: (mu+k*si)")
    # --- local pruning
    decode2.add_argument('--pr_local_expand', type=int, default=100, help="Most expansions for each state.")
    decode2.add_argument('--pr_local_diff', type=float, default=100., help="Local pruning diff/thres (1/e**D if transferring to prob.)")
    decode2.add_argument('--pr_local_penalty', type=float, default=0., help="penalize candidates from the same state.")
    # -- global pruning ("currently only tailing n-grams")
    decode2.add_argument('--pr_global_expand', type=int, default=100, help="How many states could survive in one global-beam.")
    decode2.add_argument('--pr_global_diff', type=float, default=100., help="Global pruning normalized diff/thres (1/e**D if transferring to prob.)")
    decode2.add_argument('--pr_global_penalty', type=float, default=0., help="penalize candidates from the same sig.")
    ## TODO(CHANGE): should be changed for the same for t2_*
    decode2.add_argument('--pr_global_nalpha', type=float, default=1.0, help="Length normalizer for combining.")
    decode2.add_argument('--pr_global_lreward', type=float, default=0.0, help="Length reward for combining.")
    # --- specific tailing ngram params
    decode2.add_argument('--pr_tngram_n', type=int, default=5, help="Nth tailing ngram sig for pruning.")
    decode2.add_argument('--pr_tngram_range', type=int, default=0, help="Number of the range of history for tngram, 0 for none.")
    #
    decode2.add_argument('--branching_fullfill_ratio', type=float, default=1., help="How many states to visit according to the length of first greedy one.")
    decode2.add_argument('--branching_criterion', type=str, default="abs", choices=["abs", "rel", "b_abs", "b_rel"],
                         help="When select next branches, based on which criterion.")
    decode2.add_argument('--branching_expand_run', type=int, default=1, help="How many runs to consider other branchings on later paths.")

    # cov: specific about coverage or similarities between hidden layers
    cov = parser.add_argument_group('options about cov')
    cov.add_argument('--cov_record_mode', type=str, default="none", choices=["none","max","sum"], help="heuristics about how to record cov.")
    cov.add_argument('--cov_l1_thresh', type=float, default=0.1, help="Considered as match if L1-distance <= this.")
    cov.add_argument('--cov_upper_bound', type=int, default=1, help="Upper bound for accumulated cov-att.")
    cov.add_argument('--cov_average', action='store_true', help="Average att by length.")
    #
    cov.add_argument('--hid_sim_metric', type=str, default="none", choices=["none","cos","c1","c2"], help="metrics of deciding similarities of hiddens.")
    cov.add_argument('--hid_sim_thresh', type=float, default=1.0, help="Threshold for sim of hidden vectors.")
    #
    cov.add_argument('--merge_diff_metric', type=str, default="none", choices=["none", "med"], help="Another metric for diff of merged points.")
    cov.add_argument('--merge_diff_thresh', type=int, default=100, help="Threshold for diff of merged points.")
    # specific for re-ranking & analyzing
    if phase == "rerank":
        rerank = parser.add_argument_group('options for reranking and analysing')
        #
        rerank.add_argument('--rr_mode', type=str, default="rerank", choices=["rerank", "gold"], help="Reranking mode.")
        rerank.add_argument('--rr_analysis_kbests', type=int, default=[10,1], nargs="+", help="Analysing kbests")

    # some extras
    extras = parser.add_argument_group('Extra options for comparisons.')
    # - raml
    extras.add_argument('--raml_samples', type=int, default=0, metavar='INT',
                          help="augment outputs with n samples, 0 as turned off. (default: %(default)s)")
    extras.add_argument('--raml_tau', type=float, default=0.85, metavar='FLOAT',
                          help="temperature for sharpness of exponentiated payoff distribution (default: %(default)s)")
    extras.add_argument('--raml_reward', type=str, default='ed', metavar='STR',  choices=["hd", "ed"],
                          help="reward for sampling from exponentiated payoff distribution (default: %(default)s)")
    # scheduled-sampling
    extras.add_argument('--ss_mode', type=str, default="none", choices=["none", "linear", "exp", "isigm"])
    extras.add_argument('--ss_min', type=float, default=0.5, help="Min value for ss.")
    extras.add_argument('--ss_size', type=int, default=1, help="Select s-best to sample.")
    extras.add_argument('--ss_start', type=int, default=0, help="Start uidx of ss.")
    extras.add_argument('--ss_scale', type=int, default=1000, help="Scale for uidx.")
    extras.add_argument('--ss_k', type=float, default=1.0, help="Various meanings for various ss-modes (1-ki, k^i, k/(k+e^i/k)).")

    a = parser.parse_args()

    # check options and some processing
    args = vars(a)
    check_options(args, phase)
    # init them
    zl.init_all(args)
    return args

def check_options(args, phase):
    def _warn_ifnot(ff, ss):
        if not ff:
            print("Warning for %s" % ss)
    # network
    assert args["enc_depth"] >= 1
    assert args["dec_depth"] >= 1
    # defaults
    for prefix in ["hidden_", "idrop_", "gdrop_"]:
        for n in ["dec", "enc"]:
            n0, n1 = prefix+n, prefix+"rec"
            if args[n0] is None:
                args[n0] = args[n1]
    # validation
    VALID_OPTIONS = ["ll", "bleu", "len"]
    s = args["valid_metrics"].split(",")
    assert all([one in VALID_OPTIONS for one in s])
    args["valid_metrics"] = s
    # about length
    if args["normalize_way"] == "xgaussian":
        assert args["train_len_xadd"]
    # train
    if phase == "train":
        if args["train_r2l"]:
            # assert args["decode_output_r2l"]
            _warn_ifnot(args["decode_output_r2l"], "decode_output_r2l")
            args["decode_output_r2l"] = True
        if args["dec_type"] == "ngram":
            assert args["train_mode"] != "std"
        if args["train_local_loss"].startswith("hinge"):
            _warn_ifnot(args["no_model_softmax"], "no_model_softmax")
            args["no_model_softmax"] = True
        if args["train_mode"] == "beam":
            if args["t2_beam_loss"] == "per":
                _warn_ifnot(args["no_model_softmax"], "no_model_softmax")
                args["no_model_softmax"] = True
                _warn_ifnot(args["t2_gold_run"], "t2_gold_run")
                args["t2_gold_run"] = True
            if args["t2_err_gold_lambda"] > 0 and args["t2_err_gold_mode"] == "gold":
                _warn_ifnot(args["t2_gold_run"], "t2_gold_run")
                args["t2_gold_run"] = True
            # compare hidden layers
            if args["t2_err_gold_lambda"]>0 and (args["t2_err_gold_mode"]=="based" and args["hid_sim_metric"]!="none"):
                _warn_ifnot(args["t2_gold_run"], "t2_gold_run")
                args["t2_gold_run"] = True
        # if args["train_local_loss"] == "hinge_avg":
        #     args["dynet-mem"] = "11111"
