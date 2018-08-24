# train an NMT model

# Warning: some options like the paths (python-files, data-files) and running devices should be modified to the local versions

# train for En-De

python3 ../znmt/train.py --no_overwrite --train ../../en_de_data_z5/train.final.{en,de} --dev ../../en_de_data_z5/dev.final.{en,de} --dynet-devices GPU:0 --gdrop_rec 0.2 --idrop_embedding 0.0 --drop_hidden 0.2 --drop_embedding 0.2 --lrate 0.0001 --no_validate_freq --validate_epoch --shuffle_training_data_onceatstart --no_shuffle_training_data --patience 10 --normalize_way add --normalize_alpha 0.0

# train for Zh-En
# Warning: although only ref0 is specified, mt_eval.py will strip the last digit 0 when evaluating to evaluate on the four references!

python3 ../znmt/train.py --no_overwrite --eval_metric ibleu --train ../../zh_en_data/train.final.{zh,en} --dev ../../zh_en_data/nist_2002.{src,ref0} --dynet-devices GPU:0 --gdrop_rec 0.2 --idrop_embedding 0.0 --drop_hidden 0.2 --drop_embedding 0.2 --lrate 0.0001 --dicts_rthres 30000 --no_validate_freq --validate_epoch --shuffle_training_data_onceatstart --no_shuffle_training_data --patience 10 --normalize_way add --normalize_alpha 1.0
