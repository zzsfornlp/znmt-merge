#!/usr/bin/env bash

# ASPEC data

set -e
set -v

RUNNING_DIR="$( cd "$( dirname ${BASH_SOURCE[0]}  )" && pwd )"
source ${RUNNING_DIR}/basic.sh
shopt -s expand_aliases

alias seg-jp="${HOME_DIR}/tools/kytea-0.4.7/src/bin/kytea -model ${HOME_DIR}/tools/kytea-0.4.7/data/model.bin -notags"

CUR_DATA_DIR="${HOME_DIR}/JE_data_${DIR_SUFFIX}"
mkdir ${CUR_DATA_DIR} # should not exist previously

#### specific settings ####
SP_ORI_DIR="${HOME_DIR}/data/ASPEC-JE"
SP_TR_SIZE=1500000
#### specific settings ####

function pstep1-train
{
# get training data
cat ${SP_ORI_DIR}/train/train-{1,2,3}.txt > ${CUR_DATA_DIR}/train.txt
# split them out
head -n ${SP_TR_SIZE} < ${CUR_DATA_DIR}/train.txt | perl -ne 'chomp; @a=split/ \|\|\| /; print $a[3], "\n";' > ${CUR_DATA_DIR}/train.ja
head -n ${SP_TR_SIZE} < ${CUR_DATA_DIR}/train.txt | perl -ne 'chomp; @a=split/ \|\|\| /; print $a[4], "\n";' > ${CUR_DATA_DIR}/train.en
# (Removing date expressions at EOS in Japanese in the training and development data to reduce noise)
mv ${CUR_DATA_DIR}/train.ja ${CUR_DATA_DIR}/train.ja.org
cat ${CUR_DATA_DIR}/train.ja.org | perl -Mencoding=utf8 -pe 's/(.)［[０-９．]+］$/${1}/;' > ${CUR_DATA_DIR}/train.ja
#
wc "${CUR_DATA_DIR}/train.ja"
wc "${CUR_DATA_DIR}/train.en"
}

function pstep2-dt
{
# get dt
cat ${SP_ORI_DIR}/dev/dev.txt > ${CUR_DATA_DIR}/dev.txt
cat ${SP_ORI_DIR}/test/test.txt > ${CUR_DATA_DIR}/test.txt
for name in dev test; do
  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[2], "\n";' < ${CUR_DATA_DIR}/${name}.txt > ${CUR_DATA_DIR}/${name}.ja
  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[3], "\n";' < ${CUR_DATA_DIR}/${name}.txt > ${CUR_DATA_DIR}/${name}.en
done
# (Removing date expressions at EOS in Japanese in the training and development data to reduce noise)
mv ${CUR_DATA_DIR}/dev.ja ${CUR_DATA_DIR}/dev.ja.org
cat ${CUR_DATA_DIR}/dev.ja.org | perl -Mencoding=utf8 -pe 's/(.)［[０-９．]+］$/${1}/;' > ${CUR_DATA_DIR}/dev.ja
}

function pstep3-process
{
# 1. tokenize: *.{lang} -> *.tok.{lang}
    for file in train dev test; do
        # ja
        echo "tokenizing for ${file}.ja"
        cat ${CUR_DATA_DIR}/${file}.ja | \
            perl -Mencoding=utf8 -pe 's/　/ /g;' | \
            seg-jp | \
            #<not for kytea!!> perl -ne 'chomp; if($_ eq "EOS"){print join(" ",@b),"\n"; @b=();} else {@a=split/ /; push @b, $a[0];}' | \
            perl -pe 's/^ +//; s/ +$//; s/ +/ /g;' | \
            perl -Mencoding=utf8 -pe 'tr/\|[]/｜［］/; ' > ${CUR_DATA_DIR}/${file}.tok.ja
        # en
        echo "tokenizing for ${file}.en"
        cat ${CUR_DATA_DIR}/${file}.en | \
            perl ${RUNNING_DIR}/tools/z2h.pl | \
            tokenize -l en -threads 8 > ${CUR_DATA_DIR}/${file}.tok.en
    done

# 2. clean training set
    echo "Cleaning tarin.tok.ja/en ..."
    clean-corpus ${CUR_DATA_DIR}/train.tok ja en ${CUR_DATA_DIR}/train.tok.clean 1 80

# 3. Truecaser for English
# train truecaser
    echo "Training truecaser" ${CUR_DATA_DIR}/train.tok.clean.en
    train-truecaser -corpus ${CUR_DATA_DIR}/train.tok.clean.en -model ${CUR_DATA_DIR}/truecase-model.en
    # apply truecaser
    for f in "train.tok.clean" "dev.tok" "test.tok"; do
      echo "Truecaser $f..."
      truecaser -model ${CUR_DATA_DIR}/truecase-model.en <${CUR_DATA_DIR}/$f.en >${CUR_DATA_DIR}/$f.tc.en
      # skip for ja
      ln -s ${CUR_DATA_DIR}/$f.ja ${CUR_DATA_DIR}/$f.tc.ja
    done

# files after truecase: train.tok.clean.tc.* dev/test.tok.tc.*
}

function pstep4-bpe
{    # number of ops as $1, vocab-cut as $2, vocab-thres as $3
# bpe only for English
    echo "Bpe with OP_NUM=$1, CUT_FREQ=$2, CUT_TH=$3"
    OP_NUM=$1
    CUT_FREQ=$2
    CUT_TH=$3
# learn
    echo "Learning BPE_join with bpe_operations=${OP_NUM}. This may take a while..."
    cat ${CUR_DATA_DIR}/train.tok.clean.tc.en | python3 ${SUBWORD_DIR}/learn_bpe.py -s ${OP_NUM} > ${CUR_DATA_DIR}/bpe.${OP_NUM}
    echo "Cut dictonaries for bpe-vocab.en"
    python3 ${SUBWORD_DIR}/apply_bpe.py -c ${CUR_DATA_DIR}/bpe.${OP_NUM} < ${CUR_DATA_DIR}/train.tok.clean.tc.en | \
        python3 ${SUBWORD_DIR}/get_vocab.py > ${CUR_DATA_DIR}/bpe-vocab.en
    head -n ${CUT_FREQ} <${CUR_DATA_DIR}/bpe-vocab.en  >${CUR_DATA_DIR}/bpe-vocab.cut.en
# apply
    for f in "train.tok.clean.tc" "dev.tok.tc" "test.tok.tc"; do
        echo "BPE $f..."
        python3 ${SUBWORD_DIR}/apply_bpe.py -c ${CUR_DATA_DIR}/bpe.${OP_NUM} \
            --vocabulary ${CUR_DATA_DIR}/bpe-vocab.en --vocabulary-threshold ${CUT_TH} \
            < ${CUR_DATA_DIR}/$f.en > ${CUR_DATA_DIR}/$f.bpe.en
        # skip for ja
        ln -s ${CUR_DATA_DIR}/$f.ja ${CUR_DATA_DIR}/$f.bpe.ja
    done

# files after bpe: train.tok.clean.tc.bpe.en *.tok.tc.bpe.en
}

function pstep5-concat
{
########### -- task specific -- ###########
for lang in "ja" "en"; do
    if [ -e ${CUR_DATA_DIR}/train.final.${lang} ]; then rm ${CUR_DATA_DIR}/train.final.${lang}; fi
    ln -s ${CUR_DATA_DIR}/train.${POSTFIX_TR}.${lang} ${CUR_DATA_DIR}/train.final.${lang}
    cat ${CUR_DATA_DIR}/dev.${POSTFIX_DT}.${lang} > ${CUR_DATA_DIR}/dev.final.${lang}
    cat ${CUR_DATA_DIR}/test.${POSTFIX_DT}.${lang} > ${CUR_DATA_DIR}/test.final.${lang}
    postprocess0 < ${CUR_DATA_DIR}/dev.final.${lang} > ${CUR_DATA_DIR}/dev.final.${lang}.restore
    postprocess0 < ${CUR_DATA_DIR}/test.final.${lang} > ${CUR_DATA_DIR}/test.final.${lang}.restore
done
########### -- task specific -- ###########

# prepare dictionary for nematus
for lang in "ja" "en"; do
    echo "Build dictionary for ${lang}"
    python3 ${RUNNING_DIR}/nematus_build_vocab.py \
        < "${CUR_DATA_DIR}/train.final.${lang}" > "${CUR_DATA_DIR}/vocab.${lang}.json"
done
}
