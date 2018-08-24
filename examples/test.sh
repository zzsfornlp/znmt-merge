#!/usr/bin/env bash

# Warning: some options like the paths (python-files, data-files) and running devices and especially dictionaries and models should be modified to the local versions

# test for En-De

function test1
{
    echo "running with $1, and with extras $2"
    name=$1
    extras=$2
	rundir="../run_ende/"
	python3 ../znmt/test.py -v --report_freq 128 -o output.$1 -t ../../en_de_data_z5/test.final.{en,de.restore} -d $rundir/{"src","trg"}.v -m $rundir/zbest.model --dynet-devices GPU:0 $extras
    bash ../znmt/scripts/restore.sh <output.$1 | perl ../znmt/scripts/multi-bleu.perl ../../en_de_data_z5/test.final.de.restore
}

# test for Zh-En

function test2
{
    echo "running with $1, and with extras $2"
    name=$1
    extras=$2
	rundir="../run_zhen/"
	python3 ../znmt/test.py -v --report_freq 128 --eval_metric ibleu -o output.$1 -t ../../zh_en_data/nist_36.{src,ref0} -d $rundir/{"src","trg"}.v -m $rundir/zbest.model --dynet-devices GPU:0 $extras
    perl ../znmt/scripts/multi-bleu.perl ../../zh_en_data/nist_36.ref < output.$1
}

# En-De
# run without merge
run d0 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0"
# run with merge
run z1 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 0.4 --decode_latnbest --decode_latnbest_lreward 0.4"

# Zh-En
# run without merge
run z0 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0"
# run with merge
run z1 "--test_batch_size 1 --beam_size 10 --pr_local_diff 2.3 --normalize_way norm --normalize_alpha 1.0 --pr_global_expand 1 --pr_tngram_range 2 --pr_tngram_n 4 --pr_global_nalpha 0.0 --pr_global_lreward 1.0 --decode_latnbest --decode_latnbest_lreward 1.0"

