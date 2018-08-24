#!/usr/bin/env bash

# restore from some special processing

#set -v

# postprocess0
RUNNING_DIR="$( cd "$( dirname ${BASH_SOURCE[0]}  )" && pwd )"
sed -r 's/(@@ )|(@@ ?$)//g' | perl ${RUNNING_DIR}/moses/detruecase.perl
