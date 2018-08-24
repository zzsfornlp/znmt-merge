#!/usr/bin/env bash

# download all the files needed


set -e
RUNNING_DIR="$( cd "$( dirname ${BASH_SOURCE[0]}  )" && pwd )"
source ${RUNNING_DIR}/basic.sh
shopt -s expand_aliases
set -v

# some libs needed
# -- moses
if [ ! -d ${MOSES_DIR} ]; then
  git clone https://github.com/moses-smt/mosesdecoder.git ${MOSES_DIR}
fi
# -- subword-nmt
if [ ! -d ${SUBWORD_DIR} ]; then
  git clone https://github.com/rsennrich/subword-nmt.git ${SUBWORD_DIR}
fi
# -- nematus
if [ ! -d ${NEMATUS_DIR} ]; then
  git clone https://github.com/EdinburghNLP/nematus.git ${NEMATUS_DIR}
fi

# some basic data (from wmt)
mkdir -p ${DATA_DIR}
cd ${DATA_DIR}
## ============ en-de (wmt17) ============ ##
# -- Europarl v7
echo "Downloading Europarl v7. This may take a while..."
wget -nc http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
# -- Common Crawl
echo "Downloading Common Crawl corpus. This may take a while..."
wget -nc http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
# -- News Commentary v12
echo "Downloading News Commentary v12. This may take a while..."
wget -nc http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz
## ============ en-fr (wmt14) ============ ##
# (from ~schwenk, All data originates from http://www.statmt.org/wmt14/translation-task.html)
echo "Downloading schwenk's en-fr bitext. This may take a while..."
wget -nc http://www-lium.univ-lemans.fr/~schwenk/nnmt-shared-task/data/bitexts.tgz
wget -nc http://www-lium.univ-lemans.fr/~schwenk/nnmt-shared-task/data/lm_selected.tgz
## ============ test (wmt17)  ============ ##
echo "Downloading dev/test sets"
wget -nc http://data.statmt.org/wmt17/translation-task/dev.tgz
wget -nc http://data.statmt.org/wmt17/translation-task/test.tgz

# Extract everything
echo "Extracting all files under ${DATA_DIR}"
mkdir -p "europarl-v7"
tar -xvzf "training-parallel-europarl-v7.tgz" -C "europarl-v7"
mkdir -p "commoncrawl"
tar -xvzf "training-parallel-commoncrawl.tgz" -C "commoncrawl"
mkdir -p "nc-v12"
tar -xvzf "training-parallel-nc-v12.tgz" -C "nc-v12"
mkdir -p "bitexts"
tar -xvzf "bitexts.tgz" -C "bitexts"
mkdir -p "lm_selected"
tar -xvzf "lm_selected.tgz" -C "lm_selected"
mkdir -p "dev"
tar -xvzf "dev.tgz" -C "dev"
mkdir -p "test"
tar -xvzf "test.tgz" -C "test"
