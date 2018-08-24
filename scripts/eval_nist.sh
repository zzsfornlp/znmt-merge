#!/usr/bin/env bash

# restore from some special processing

#set -v

# usage (need to specify much more)
echo "Running with lang: $1, src: $2, ref: $3, trg: $4, lang-full: $5"

RUNNING_DIR="$( cd "$( dirname ${BASH_SOURCE[0]}  )" && pwd )"
sed -r 's/(@@ )|(@@ ?$)//g' <$4 | \
    perl ${RUNNING_DIR}/moses/detruecase.perl | \
    perl ${RUNNING_DIR}/moses/detokenizer.perl -l $1 | \
    perl ${RUNNING_DIR}/moses/wrap-xml.perl $5 $2 znmt > $4.sgm
perl ${RUNNING_DIR}/moses/mteval-v13a.pl -s $2 -r $3 -t $4.sgm
