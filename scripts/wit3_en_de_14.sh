#!/usr/bin/env bash

# (for IWSLT & WIT3)
# special en-de data for iwslt14
# todo(warn): directly call without process.sh

set -e
set -v

MOSES_DIR="../../mosesdecoder"
alias tokenize="perl ${MOSES_DIR}/scripts/tokenizer/tokenizer.perl"
alias clean-corpus="perl ${MOSES_DIR}/scripts/training/clean-corpus-n.perl"
alias lowercaser="perl ${MOSES_DIR}/scripts/tokenizer/lowercase.perl"
shopt -s expand_aliases

# prepared already, inside en-de_2014
SRC="en"
TRG="de"

function pstep1-train
{
if [ -r train.${TRG} ]; then mv train.${TRG} nothing.${TRG}; fi
# -- train: delete tags
sed "/^[ \t]*</d" < train.tags.${SRC}-${TRG}.${SRC} > train.${SRC}
sed "/^[ \t]*</d" < train.tags.${SRC}-${TRG}.${TRG} > train.${TRG}
}

function pstep2-dt
{
# -- dev/test: extract from xml
for f in *.xml; do
    sed -r 's/<seg id=.*>(.*)<\/seg>/\1/g' < $f | sed "/^[ \t]*</d" > dt.${f%.*};
done
}

function pstep3-process
{
    # special preparing
    DIR="."
    # 1. Tokenize data + lowercase
    for lang in ${SRC} ${TRG}; do
        for f in ${DIR}/train.${lang} ${DIR}/dt.*.${lang}; do
          echo "Tokenizing $f with lowercase ..."
          tokenize -l ${lang} -threads 8 < $f | lowercaser > ${f%.*}.tok.${lang}
        done
    done
    # 2. clean training data
    # Clean training set
    for f in ${DIR}/train.tok.${SRC}; do
      fbase=${f%.*}
      echo "Cleaning ${fbase}..."
      clean-corpus ${fbase} ${SRC} ${TRG} ${fbase}.clean 1 50
    done
}

# should be step4 though
function pstep5-concat
{
########### -- task specific -- ###########
for lang in ${SRC} ${TRG}; do
    head -153000 train.tok.clean.${lang} > train.final.${lang}
    tail -6969 train.tok.clean.${lang} > dev.final.${lang}
#    # according to harvardnlp-BSO's preparation script
#    awk '{if (NR%23 == 0)  print $0; }' train.tok.clean.${lang} > dev.final.${lang}
#    awk '{if (NR%23 != 0)  print $0; }' train.tok.clean.${lang} > train.final.${lang}
    cat dt.*.tok.${lang} > test.final.${lang}
done
########### -- task specific -- ###########

# prepare dictionary for nematus
for lang in ${SRC} ${TRG}; do
    echo "Build dictionary for ${lang}"
    python3 ../../znmt/scripts/nematus_build_vocab.py \
        < "train.final.${lang}" > "vocab.${lang}.json"
done
}

time pstep1-train
time pstep2-dt
time pstep3-process
time pstep5-concat
