#!/usr/bin/env bash

# FR && EN

set -e
set -v

RUNNING_DIR="$( cd "$( dirname ${BASH_SOURCE[0]}  )" && pwd )"
source ${RUNNING_DIR}/basic.sh
shopt -s expand_aliases

SRC="en"
TRG="fr"

CUR_DATA_DIR="${HOME_DIR}/${SRC}_${TRG}_data_${DIR_SUFFIX}"
mkdir ${CUR_DATA_DIR} # should not exist previously

function pstep1-train
{
# Concatenate training data (12M)
zcat ${DATA_DIR}/bitexts/bitexts.selected/*.en.gz > "${CUR_DATA_DIR}/train.en"
wc "${CUR_DATA_DIR}/train.en"
zcat ${DATA_DIR}/bitexts/bitexts.selected/*.fr.gz > "${CUR_DATA_DIR}/train.fr"
wc "${CUR_DATA_DIR}/train.fr"
}

function pstep2-dt
{
# get dev and test (start with news*)
cp ${DATA_DIR}/dev/dev/newstest2012.{${SRC},${TRG}} ${CUR_DATA_DIR}
cp ${DATA_DIR}/dev/dev/newstest2013.{${SRC},${TRG}} ${CUR_DATA_DIR}
input-from-sgm <"${DATA_DIR}/dev/dev/newstest2014-fren-src.${SRC}.sgm" >"${CUR_DATA_DIR}/newstest2014.${SRC}"
input-from-sgm <"${DATA_DIR}/dev/dev/newstest2014-fren-ref.${TRG}.sgm" >"${CUR_DATA_DIR}/newstest2014.${TRG}"
input-from-sgm <"${DATA_DIR}/dev/dev/newsdiscussdev2015-${SRC}${TRG}-src.${SRC}.sgm" >"${CUR_DATA_DIR}/newsdiscussdev2015.${SRC}"
input-from-sgm <"${DATA_DIR}/dev/dev/newsdiscussdev2015-${SRC}${TRG}-ref.${TRG}.sgm" >"${CUR_DATA_DIR}/newsdiscussdev2015.${TRG}"
input-from-sgm <"${DATA_DIR}/dev/dev/newsdiscusstest2015-${SRC}${TRG}-src.${SRC}.sgm" >"${CUR_DATA_DIR}/newsdiscusstest2015.${SRC}"
input-from-sgm <"${DATA_DIR}/dev/dev/newsdiscusstest2015-${SRC}${TRG}-ref.${TRG}.sgm" >"${CUR_DATA_DIR}/newsdiscusstest2015.${TRG}"
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
