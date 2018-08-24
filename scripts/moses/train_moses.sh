#!/usr/bin/env bash

# according to http://lotus.kuee.kyoto-u.ac.jp/WAT/WAT2017/baseline/baselineSystemPhrase.html

# compile moses & mgiza
# ./bjam --with-boost=../../libs/boost_1_60_0/ -j20
# mgiza no static compiling

# Setup
LANG_F=zh
LANG_E=en
CORPUS_LM=`pwd`/../zh_en_data/train.final
CORPUS=`pwd`/../zh_en_data/train.final
DEV_F=`pwd`/../zh_en_data/Dev-set/nist_2002.src
DEV_E=`pwd`/../zh_en_data/Dev-set/nist_2002.ref
TEST=`pwd`/../zh_en_data/Dev-set/nist_2002.src
REF=`pwd`/../zh_en_data/Dev-set/nist_2002.ref
LM_ORDER=5
JOBS=16

MOSES_SCRIPT=`pwd`/../mosesdecoder/scripts
MOSES_BIN_DIR=`pwd`/../mosesdecoder/bin
#EXT_BIN_DIR=`pwd`/../giza-pp/GIZA++-v2/
EXT_BIN_DIR=`pwd`/../giza-pp/mgiza/mgizapp/build/

WORK_DIR=work.${LANG_F}-${LANG_E}
TRAINING_DIR=${WORK_DIR}/training
MODEL_DIR=${WORK_DIR}/training/model

mkdir phraseModel
cd phraseModel/
mkdir -p ${TRAINING_DIR}/lm

# LM
LM_FILE=`pwd`/${TRAINING_DIR}/lm/lm.${LANG_E}.arpa.gz
${MOSES_BIN_DIR}/lmplz --order ${LM_ORDER} -S 80% -T /tmp < ${CORPUS_LM}.${LANG_E} | gzip > ${LM_FILE}

# translation model
perl ${MOSES_SCRIPT}/training/train-model.perl \
  --root-dir `pwd`/${TRAINING_DIR} \
  --model-dir `pwd`/${MODEL_DIR} \
  --corpus ${CORPUS} \
  --external-bin-dir ${EXT_BIN_DIR} \
  --f ${LANG_F} \
  --e ${LANG_E} \
  --parallel \
  --alignment grow-diag-final-and \
  --reordering msd-bidirectional-fe \
  --score-options "--GoodTuring" \
  --lm 0:${LM_ORDER}:${LM_FILE}:8 \
  --cores ${JOBS} \
  --sort-buffer-size 10G \
  --parallel \
  -mgiza\
  -mgiza-cpus ${JOBS} \
  >& ${TRAINING_DIR}/training_TM.log

# tuning
perl ${MOSES_SCRIPT}/training/filter-model-given-input.pl \
  ${MODEL_DIR}.filtered/dev \
  ${MODEL_DIR}/moses.ini \
  ${DEV_F}

mkdir -p ${WORK_DIR}/tuning

perl ${MOSES_SCRIPT}/training/mert-moses.pl \
  ${DEV_F} \
  ${DEV_E} \
  ${MOSES_BIN_DIR}/moses \
  `pwd`/${MODEL_DIR}.filtered/dev/moses.ini \
  --mertdir ${MOSES_BIN_DIR} \
  --working-dir `pwd`/${WORK_DIR}/tuning/mert \
  --threads ${JOBS} \
  --no-filter-phrase-table \
  --decoder-flags "-threads ${JOBS} -distortion-limit 20" \
  --predictable-seeds \
  >& ${WORK_DIR}/tuning/mert.log


perl ${MOSES_SCRIPT}/ems/support/substitute-weights.perl \
  ${MODEL_DIR}/moses.ini \
  ${WORK_DIR}/tuning/mert/moses.ini \
  ${MODEL_DIR}/moses-tuned.ini

# testing
OUTPUT_DIR=${WORK_DIR}/output
mkdir -p ${OUTPUT_DIR}

perl ${MOSES_SCRIPT}/training/filter-model-given-input.pl \
  ${MODEL_DIR}.filtered/test \
  ${MODEL_DIR}/moses-tuned.ini \
  ${TEST}

outfile=${OUTPUT_DIR}/test.out

${MOSES_BIN_DIR}/moses -config ${MODEL_DIR}.filtered/test/moses.ini -distortion-limit 20 -threads ${JOBS} < ${TEST} > ${outfile} 2> ${outfile}.log
