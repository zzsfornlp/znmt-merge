#!/usr/bin/env bash

# (for IWSLT & WIT3)

set -e
set -v

RUNNING_DIR="$( cd "$( dirname ${BASH_SOURCE[0]}  )" && pwd )"
source ${RUNNING_DIR}/basic.sh
shopt -s expand_aliases

mkdir -p ${DATA2_DIR}
cd ${DATA2_DIR}

# !-- pass as env
#SRC="en"
#TRG="fr"

# get data (under the dir of data2)
data_name="wit3-${SRC}-${TRG}_${DIR_SUFFIX}"
wget -nc https://wit3.fbk.eu/archive/2017-01-trnted//texts/${SRC}/${TRG}/${SRC}-${TRG}.tgz -O "${data_name}.tgz"
# special treating
mkdir ${data_name}
tar -zxvf "${data_name}.tgz" -C ${data_name}
cd ${data_name}/"${SRC}-${TRG}"
mv * ..
cd ..
rmdir "${SRC}-${TRG}"

CUR_DATA_DIR="${DATA2_DIR}/${data_name}"

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
    cat ${CUR_DATA_DIR}/dt.*{2013,2014}*.${POSTFIX_DT}.${lang} > ${CUR_DATA_DIR}/dev.final.${lang}
    cat ${CUR_DATA_DIR}/dt.*2015*.${POSTFIX_DT}.${lang} > ${CUR_DATA_DIR}/test.final.${lang}
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