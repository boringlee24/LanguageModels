#!/bin/bash

if [ $(whoami) == JO30252 ]; then
  export TMPDIR="/home/gridsan/$(whoami)/slurm_tmp/${SLURM_JOBID}"
fi

LOG_DIR=$1
NODE_NAME=$(hostname -s)
export DCGM_LOG=${LOG_DIR}/dcgm-${NODE_NAME}.log

# DISCONTINUE DCGM USE BECAUSE OF FAULTY MEASUREMENT
# # stop collection and show stats
# echo "stopping NVIDIA DCGM stat collection for job ${SLURM_JOBID}"
# dcgmi stats -x $SLURM_JOBID -v
# # show stats : 
# echo "writing stats to file : ${DCGM_LOG}"
# echo "NVIDIA DCGM stats for job ${SLURM_JOBID} : "
# dcgmi stats -j $SLURM_JOBID -v 2>&1 > ${DCGM_LOG}

# COMMENT THE BELOW LINES ON E1, UNCOMMENT ON GREEN
# # stop nvidia-smi logging
# PID=$(pidof nvidia-smi)
# echo "Still the PID of profile script : ${PID}"
# kill -9 ${PID}
# mv -v ${TMPDIR}/${SLURM_JOBID}-$(uname -n)-nvidia-smi.csv ${LOG_DIR}/

# RESTORE DEFAULT POWER LIMIT
sidhelper 250
