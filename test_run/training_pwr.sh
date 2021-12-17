#!/bin/bash
echo "run nvidia-smi command to monitor gpu power"

RUN_TIME=1800 #1800
TESTCASE="bert"
GPU=$1 # v100
DATA_PATH="/scratch/$(whoami)/bert-mlm/test_dir/"

mkdir -p ${DATA_PATH}

timeout ${RUN_TIME} nvidia-smi -i 0 --query-gpu=index,timestamp,power.draw,memory.used,utilization.memory,utilization.gpu,temperature.gpu,temperature.memory,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,clocks.gr,clocks.sm,clocks.mem --format=csv,nounits -lms 200 --filename=${DATA_PATH}${GPU}_${TESTCASE}.csv

