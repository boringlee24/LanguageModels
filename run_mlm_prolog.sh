#!/bin/bash

if [ $(whoami) == JO30252 ]; then
  export TMPDIR="/home/gridsan/$(whoami)/slurm_tmp/${SLURM_JOBID}"
  mkdir -pv $TMPDIR
fi

# set power limit
echo "setting power limit"
POWER_LIMIT=$1
sidhelper ${POWER_LIMIT}

# NO MORE DCGM MONITORING SINCE DISCOVERING ISSUE
# # setup NVIDIA monitoring
# echo "starting GPU monitoring for job ${SLURM_JOBID}"
# # dcgmi group -c allgpus --default  # UNCOMMENT ON TXG, COMMENT ON E1
# # dcgmi group -c ${SLURM_JOB_ID} -a $CUDA_VISIBLE_DEVICES  # SAME AS ABOVE
# dcgmi stats -g 2 -e
# dcgmi stats -g 2 -s $SLURM_JOBID -v

# These lines will setup nvidia-smi monitoring. We don't monitor jobs on tx-green, so we have to do this ourselves.
# COMMENT THE BELOW LINES ON E1
# echo "nvidia-smi file : ${TMPDIR}/${SLURM_JOBID}-$(uname -n)-nvidia-smi.csv"
# CVD=$(tr -dc '[:alnum:]-,' <<< $CUDA_VISIBLE_DEVICES)
# (${HOME}/bin/nvidia-smi-profile.sh ${CVD} ${TMPDIR}/${SLURM_JOBID}-$(uname -n)-nvidia-smi.csv & )
# PID=$(pidof nvidia-smi)
# echo "PID of profile script : ${PID}"
