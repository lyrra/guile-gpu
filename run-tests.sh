#!/bin/sh
#
# args:
#   --gpu  -- Use gpu
#

. ./env.sh

C="guile $GUILE_CODE_LOAD_PATH run-tests.scm $*"
echo $C
$C

