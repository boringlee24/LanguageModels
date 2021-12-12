#!/bin/bash
pip install --user transformers datasets
# different group directory on txe vs txg
if [ $(whoami) == jpmcd ]
then
    GROUP_DIR="fastai"
else
    GROUP_DIR="fastai_shared"
fi
mkdir -p /home/gridsan/groups/${GROUP_DIR}/$(whoami)/fastai/ground-truth-logs/slurm_logs/
mkdir -p /home/gridsan/groups/${GROUP_DIR}/$(whoami)/fastai/ground-truth-logs/logs/
