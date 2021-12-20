#!/bin/bash

#module load cuda/10.1

POWER=$1

export HF_USER_DIR="/home/$(whoami)/.cache/huggingface"
export HF_LOCAL_DIR="/home/$(whoami)/.cache/huggingface"
mkdir -p $HF_LOCAL_DIR
export HF_HOME=${HF_LOCAL_DIR}
#export TRANSFORMERS_OFFLINE=1
#export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED="true"

export MODEL_TYPE="bert"
export TOKENIZER_NAME="bert-base-uncased"
export DATASET_NAME="wikitext"
export DATASET_CONFIG="wikitext-2-raw-v1"
#export DATASET_CONFIG="wikitext-103-raw-v1"
BATCH_SIZE=8
NUM_EPOCHS=2.0

export OUTPUT_DIR="/home/$(whoami)/bert-mlm/test_dir/${POWER}"

rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

sudo nvidia-smi -i 0 -pm 1 #TODO

sudo nvidia-smi -i 0 -pl $POWER #TODO

./training_pwr.sh "k80_${POWER}" & 

python run_mlm.py \
    --model_type ${MODEL_TYPE} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --output_dir ${OUTPUT_DIR}


