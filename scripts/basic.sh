#!/usr/bin/env bash

# set up the basic directory

#set -e
#set -v
shopt -s expand_aliases

# some settings
if [ -n "${ZMT}" ];
then HOME_DIR="${ZMT}";
else HOME_DIR=`pwd`;
fi
DATA_DIR="${HOME_DIR}/data"
MOSES_DIR="${HOME_DIR}/mosesdecoder"
SUBWORD_DIR="${HOME_DIR}/subword-nmt"
NEMATUS_DIR="${HOME_DIR}/nematus"
#echo "choosing current dir" $HOME_DIR "as the home-dir for the mt-basic." 1>&2
DATA2_DIR="${HOME_DIR}/data2"   # WIT3

# tools from moses
alias input-from-sgm="perl ${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl"
alias tokenize="perl ${MOSES_DIR}/scripts/tokenizer/tokenizer.perl"
alias clean-corpus="perl ${MOSES_DIR}/scripts/training/clean-corpus-n.perl"
alias train-truecaser="perl ${MOSES_DIR}/scripts/recaser/train-truecaser.perl"
alias truecaser="perl ${MOSES_DIR}/scripts/recaser/truecase.perl"
alias lowercaser="perl ${MOSES_DIR}/scripts/tokenizer/lowercase.perl"
alias multi-bleu="perl ${MOSES_DIR}/scripts/generic/multi-bleu.perl"

# prepare-data <DIR> <SRC> <TRG>; at lease for en-fr en-de
# preparing for $DIR/(train/dev/test).(${SRC}/${TRG})
# -> train.{lang}, dt.*.{lang}
function prepare-data
{
    # check
    echo "Preparing data with DIR=$1, SRC=$2, TRG=$3"
    DIR=$1
    SRC=$2
    TRG=$3
    if [ -z "${DIR}" -o -z "${SRC}" -o -z "${TRG}" ];
    then echo "Parameters not specified for prepare-data."; return 0;
    fi

    # Tokenize data
    for lang in ${SRC} ${TRG}; do
        for f in ${DIR}/train.${lang} ${DIR}/dt.*.${lang}; do
          echo "Tokenizing $f..."
          tokenize -l ${lang} -threads 8 < $f > ${f%.*}.tok.${lang}
        done
    done

    # Clean training set
    for f in ${DIR}/train.tok.${SRC}; do
      fbase=${f%.*}
      echo "Cleaning ${fbase}..."
      clean-corpus ${fbase} ${SRC} ${TRG} ${fbase}.clean 1 80
    done

    # Truecaser
    # train truecaser
    for lang in ${SRC} ${TRG}; do
        echo "Training truecaser" ${DIR}/train.tok.clean.${lang}
        train-truecaser -corpus ${DIR}/train.tok.clean.${lang} -model ${DIR}/truecase-model.${lang}
        # apply truecaser
        for f in ${DIR}/train.tok.clean.${lang} ${DIR}/dt.*.tok.${lang}; do
          echo "Truecaser $f..."
          truecaser -model ${DIR}/truecase-model.${lang} <$f >${f%.*}.tc.${lang}
        done
    done

    # files after truecase: train.tok.clean.tc.en *.tok.tc.en
}

# tc -> bpe (simply set $2 or $3 == "" will apply bpe for single language)
function bpe
{
    echo "Bpe with DIR=$1, SRC=$2, TRG=$3, OP_NUM=$4"
    DIR=$1
    SRC=$2
    TRG=$3
    OP_NUM=$4
    # learn
    echo "Learning BPE with bpe_operations=${OP_NUM}. This may take a while..."
    cat ${DIR}/train.tok.clean.tc.${SRC} ${DIR}/train.tok.clean.tc.${TRG}| \
        python3 ${SUBWORD_DIR}/learn_bpe.py -s ${OP_NUM} > ${DIR}/bpe.${OP_NUM}
    # apply
    for lang in ${SRC} ${TRG}; do
        for f in ${DIR}/train.tok.clean.tc.${lang} ${DIR}/dt.*.tok.tc.${lang}; do
            echo "Apply BPE with bpe_operations=${OP_NUM} to $f"
            outfile="${f%.*}.bpe.${lang}"
            python3 ${SUBWORD_DIR}/apply_bpe.py -c ${DIR}/bpe.${OP_NUM} < $f > ${outfile}
        done
    done

    # files after bpe: train.tok.clean.tc.bpe.en *.tok.tc.bpe.en
}

# better practice for join bpe (tc->bpe)
function bpe-join
{
    echo "Bpe with DIR=$1, SRC=$2, TRG=$3, OP_NUM=$4, CUT_FREQ=$5, CUT_TH=$6"
    DIR=$1
    SRC=$2
    TRG=$3
    OP_NUM=$4
    CUT_FREQ=$5
    CUT_TH=$6
    # learn
    echo "Learning BPE_join with bpe_operations=${OP_NUM}. This may take a while..."
    python3 ${SUBWORD_DIR}/learn_joint_bpe_and_vocab.py \
        --input ${DIR}/train.tok.clean.tc.${SRC} ${DIR}/train.tok.clean.tc.${TRG} -s ${OP_NUM} -o ${DIR}/bpe.${OP_NUM} \
        --write-vocabulary ${DIR}/bpe-vocab.${SRC} ${DIR}/bpe-vocab.${TRG}
    # apply
    for lang in ${SRC} ${TRG}; do
        echo "Cut dictonaries for bpe-vocab.${lang}"
        head -n ${CUT_FREQ} <${DIR}/bpe-vocab.${lang}  >${DIR}/bpe-vocab.cut.${lang}
        for f in ${DIR}/train.tok.clean.tc.${lang} ${DIR}/dt.*.tok.tc.${lang}; do
            echo "Apply BPE with bpe_operations=${OP_NUM} to $f"
            outfile="${f%.*}.bpe.${lang}"
            python3 ${SUBWORD_DIR}/apply_bpe.py -c ${DIR}/bpe.${OP_NUM} \
                --vocabulary ${DIR}/bpe-vocab.cut.${lang} --vocabulary-threshold ${CUT_TH} < $f > ${outfile}
        done
    done

    # files after bpe: train.tok.clean.tc.bpe.en *.tok.tc.bpe.en
}

# restore the original one -- the sed will also work for non-bpe assuming @@-uffix is unseen
# preparing for $LANG, and io
function postprocess
{
    sed -r 's/(@@ )|(@@ ?$)//g' | \
    perl ${MOSES_DIR}/scripts/recaser/detruecase.perl | \
    perl ${MOSES_DIR}/scripts/tokenizer/detokenizer.perl -l $1
}
# no detokenizer version of postprocess
function postprocess0
{
    sed -r 's/(@@ )|(@@ ?$)//g' | \
    perl ${MOSES_DIR}/scripts/recaser/detruecase.perl
}
