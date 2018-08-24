There are many cmd options (feel free to explore them in `ztry1/mt_args.py`), we will describe the ones related with decoding here in more details.

* Basic:
* `--beam_size/-k`: beam size
* `--decode_len`: maximum decode length, force stop after this number of steps
* `--decode_ratio`: maximum decode length = (source-length * decode_ratio)
* `--eval_metric`: case-sensitive (bleu) or -insensitive BLEU (ibleu)
* `--test_batch_size`: batch size when decoding, usually train-batch-size/beam-size
* `--normalize_way`: length normalization (norm) or length reward (add) or some others
* `--normalize_alpha`: length normalizer for norm or length reward \lambda or others

* Pruning and Merge:
* `--pr_local_expand`: maximum number of local expansions from one previous-step step, usually beam size.
* `--pr_local_diff`: prune after the local score is this much worse than the best local one. (Can be interpretated as percentage threshold for local normalized model, for example, use 2.3, which equals to around -ln(10%), for 10%).
* `--pr_global_expand`: similar to `pr_local_expand`, but for the global (merge) pruner.
* `--pr_global_diff`: similar to `pr_local_diff`, but for the global (merge) pruner.
* `--pr_tngram_n`: n-gram suffix merger.
* `--pr_tngram_range`: length difference threshold (r) for merging.
* `--pr_global_lreward`: length reward (\lambda) for partial hypothesis comparison.
* `--pr_global_nalpha`: length normalizer for partial hypothesis comparison.
* `--decode_latnbest`: obtaining k-best list from second search in the lattice.
* `--decode_latnbest_nalpha`: length normalizer for the second search, usually same as `pr_global_nalpha`.
* `--decode_latnbest_lreward`: length reward (\lambda) for the second search, usually same as `pr_global_lreward`.

Please refer to `examples/test.sh` for running examples.
