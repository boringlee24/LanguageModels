#!/bin/bash
if [ ! -e /proc/$(pidof nvidia-smi) ]
then
	echo "nvidia-smi does not seem to be running. exiting job"
    exit 1
fi

MODEL_TYPE="bert"
TOKENIZER_NAME="bert-base-uncased"
DATASET_NAME="wikitext"
# DATASET_CONFIG="wikitext-2-raw-v1"
DATASET_CONFIG="wikitext-103-raw-v1"
BATCH_SIZE=16
NUM_EPOCHS=10.0
OUTPUT_DIR='logs'

python -m torch.distributed.launch run_mlm.py \
    --model_type ${MODEL_TYPE} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --output_dir ${OUTPUT_DIR}

echo "all done"
