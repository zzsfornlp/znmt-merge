#!/usr/bin/env bash

# DE && EN
# this file is from seq2seq: https://github.com/google/seq2seq/blob/master/bin/data/wmt16_en_de.sh
# also refer to https://github.com/rsennrich/wmt16-scripts
# -- running from pwd, treat it as home dir

set -e
set -v

RUNNING_DIR="$( cd "$( dirname ${BASH_SOURCE[0]}  )" && pwd )"
source ${RUNNING_DIR}/basic.sh
shopt -s expand_aliases

SRC="en"
TRG="de"

CUR_DATA_DIR="${HOME_DIR}/${SRC}_${TRG}_data_${DIR_SUFFIX}"
mkdir ${CUR_DATA_DIR} # should not exist previously

function pstep1-train
{
# Concatenate training data (lines: 4590101)
cat "${DATA_DIR}/europarl-v7/training/europarl-v7.de-en.en" \
  "${DATA_DIR}/commoncrawl/commoncrawl.de-en.en" \
  "${DATA_DIR}/nc-v12/training/news-commentary-v12.de-en.en" \
  > "${CUR_DATA_DIR}/train.en"
wc "${CUR_DATA_DIR}/train.en"
cat "${DATA_DIR}/europarl-v7/training/europarl-v7.de-en.de" \
  "${DATA_DIR}/commoncrawl/commoncrawl.de-en.de" \
  "${DATA_DIR}/nc-v12/training/news-commentary-v12.de-en.de" \
  > "${CUR_DATA_DIR}/train.de"
wc "${CUR_DATA_DIR}/train.de"
}

function pstep2-dt
{
# get dev and test (start with news*)
cp ${DATA_DIR}/dev/dev/newstest2012.{${SRC},${TRG}} ${CUR_DATA_DIR}
cp ${DATA_DIR}/dev/dev/newstest2013.{${SRC},${TRG}} ${CUR_DATA_DIR}
input-from-sgm <"${DATA_DIR}/dev/dev/newstest2014-deen-src.${SRC}.sgm" >"${CUR_DATA_DIR}/newstest2014.${SRC}"
input-from-sgm <"${DATA_DIR}/dev/dev/newstest2014-deen-ref.${TRG}.sgm" >"${CUR_DATA_DIR}/newstest2014.${TRG}"
input-from-sgm <"${DATA_DIR}/dev/dev/newstest2015-${SRC}${TRG}-src.${SRC}.sgm" >"${CUR_DATA_DIR}/newstest2015.${SRC}"
input-from-sgm <"${DATA_DIR}/dev/dev/newstest2015-${SRC}${TRG}-ref.${TRG}.sgm" >"${CUR_DATA_DIR}/newstest2015.${TRG}"
input-from-sgm <"${DATA_DIR}/dev/dev/newstest2016-${SRC}${TRG}-src.${SRC}.sgm" >"${CUR_DATA_DIR}/newstest2016.${SRC}"
input-from-sgm <"${DATA_DIR}/dev/dev/newstest2016-${SRC}${TRG}-ref.${TRG}.sgm" >"${CUR_DATA_DIR}/newstest2016.${TRG}"
for ff in ${CUR_DATA_DIR}/news*; do
    fname=`basename ${ff}`
    mv ${ff} ${CUR_DATA_DIR}/dt.${fname}   # dev or test
done
}

function pstep3-process
{
prepare-data "${CUR_DATA_DIR}" "${SRC}" "${TRG}"
}

function pstep4-bpe
{    # number of ops as $1, vocab-cut as $2, vocab-thres as $3
bpe-join "${CUR_DATA_DIR}" "${SRC}" "${TRG}" $1 $2 $3
}

function pstep5-concat
{
########### -- task specific -- ###########
for lang in ${SRC} ${TRG}; do
    if [ -e ${CUR_DATA_DIR}/train.final.${lang} ]; then rm ${CUR_DATA_DIR}/train.final.${lang}; fi
    ln -s ${CUR_DATA_DIR}/train.${POSTFIX_TR}.${lang} ${CUR_DATA_DIR}/train.final.${lang}
    cat ${CUR_DATA_DIR}/dt.newstest201{2,3}.${POSTFIX_DT}.${lang} > ${CUR_DATA_DIR}/dev.final.${lang}
    cat ${CUR_DATA_DIR}/dt.newstest2014.${POSTFIX_DT}.${lang} > ${CUR_DATA_DIR}/test.final.${lang}
    postprocess0 < ${CUR_DATA_DIR}/dev.final.${lang} > ${CUR_DATA_DIR}/dev.final.${lang}.restore
    postprocess0 < ${CUR_DATA_DIR}/test.final.${lang} > ${CUR_DATA_DIR}/test.final.${lang}.restore
done
########### -- task specific -- ###########

# prepare dictionary for nematus
for lang in ${SRC} ${TRG}; do
    echo "Build dictionary for ${lang}"
    python3 ${RUNNING_DIR}/nematus_build_vocab.py \
        < "${CUR_DATA_DIR}/train.final.${lang}" > "${CUR_DATA_DIR}/vocab.${lang}.json"
done
}
