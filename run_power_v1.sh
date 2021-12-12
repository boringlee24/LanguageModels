#!/bin/bash

if [ $(whoami) == jpmcd ]; then
  GROUP_DIR="fastai"
else
  GROUP_DIR="fastai_shared"
fi

# Make Hugging Face cache folder on drive
HF_LOCAL_DIR="/state/partition1/user/$(whoami)/cache/huggingface"
mkdir -p $HF_LOCAL_DIR
# HF folder in shared file system
HF_USER_DIR="/home/gridsan/$(whoami)/.cache/huggingface"
export HF_HOME="${HF_LOCAL_DIR}"

# general job parameters
SLURM_LOG_DIR="/home/gridsan/$(whoami)/greenai/slurm_logs"
POWER_LIMIT=250  # minimum is 100W, max. is 250W maybe set at command line
EXCLUDED="c-9-12-1,c-7-12-2"

# hugging face parameters
HF_MODEL_NAME="bert-base-uncased"  # default "distilbert-base-uncased"
GLUE_TASK="sst2"  # default "sst2"
BATCH_SIZE=32
LEARNING_RATE="2e-5"
WEIGHT_DECAY="0.01"
NUM_EPOCHS=5

# python load_datasets.py
python load_model.py --model ${HF_MODEL_NAME}
rsync -a --ignore-existing ${HF_LOCAL_DIR}/ $HF_USER_DIR
for i in `seq 1 1`;
do
    sbatch --output="${SLURM_LOG_DIR}/%j.log" \
        -N 1 \
        --exclude=${EXCLUDED} \
        batch_power_v1.sh ${HF_MODEL_NAME} \
        ${GLUE_TASK} \
        ${BATCH_SIZE} \
        ${LEARNING_RATE} \
        ${WEIGHT_DECAY} \
        ${NUM_EPOCHS} \
        ${POWER_LIMIT}
done
