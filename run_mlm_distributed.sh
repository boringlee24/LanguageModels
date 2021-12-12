#!/bin/bash
#SBATCH -N 8
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH -o /home/gridsan/%u/languagemodels/bert-mlm/slurm-%j.log
##SBATCH --mail-user=jpmcd@mit.edu
#SBATCH --mail-type=END,FAIL
##SBATCH --reservation=PowerTesting
##SBATCH --reservation=DCGM-Testing

if [ ! -e /proc/$(pidof nvidia-smi) ]
then
	echo "nvidia-smi does not seem to be running. exiting job"
    exit 1
fi

source /etc/profile
module load anaconda/2021a
module load mpi/openmpi-4.0
module load cuda/10.1
module load nccl/2.5.6-cuda10.1

START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export TOTAL_GPUS=${SLURM_NTASKS}
export GPUS_PER_NODE=2
export POWER_LIMIT=150  # minimum is 100W, max. is 250W maybe set at command line

# no change needed here : 
export MPI_FLAGS="--tag-output --bind-to socket -map-by core -mca btl ^openib -mca pml ob1 -x PSM2_GPUDIRECT=1 -x NCCL_NET_GDR_LEVEL=5 -x NCCL_P2P_LEVEL=5 -x NCCL_NET_GDR_READ=1"

# Set some environment variables needed by torch.distributed 
export MASTER_ADDR=$(hostname -s)
# Get unused port
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "START_TIME  : ${START_TIME}"
echo "MASTER_ADDR : ${MASTER_ADDR}"
echo "MASTER_PORT : ${MASTER_PORT}"

export HF_USER_DIR="/home/gridsan/$(whoami)/.cache/huggingface"
export HF_LOCAL_DIR="/state/partition1/user/$(whoami)/cache/huggingface"
# mkdir -p $HF_LOCAL_DIR
# rsync -a --ignore-existing $HF_USER_DIR/ ${HF_LOCAL_DIR}
mpirun -x HF_LOCAL_DIR --npernode 1 mkdir -pv ${HF_LOCAL_DIR}
mpirun -x HF_USER_DIR -x HF_LOCAL_DIR --npernode 1 rsync -a --ignore-existing $HF_USER_DIR/ ${HF_LOCAL_DIR}
export HF_HOME=${HF_LOCAL_DIR}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED="true"

# MODEL_TYPE="big_bird"
# TOKENIZER_NAME="google/bigbird-roberta-base"
# MODEL_TYPE="distilbert"
# TOKENIZER_NAME="distilbert-base-uncased"
MODEL_TYPE="bert"
TOKENIZER_NAME="bert-base-uncased"
DATASET_NAME="wikitext"
# DATASET_CONFIG="wikitext-2-raw-v1"
DATASET_CONFIG="wikitext-103-raw-v1"
BATCH_SIZE=16
NUM_EPOCHS=10.0

export BASEDIR="/home/gridsan/$(whoami)/languagemodels/bert-mlm"
export LOG_DIR="${BASEDIR}/${SLURM_JOBID}-${TOTAL_GPUS}-${BATCH_SIZE}-${NUM_EPOCHS}-${POWER_LIMIT}W"
mkdir -pv ${LOG_DIR}
OUTPUT_DIR="${LOG_DIR}/output"
LOG_FILE="${LOG_DIR}/${SLURM_JOBID}.log"
ERR_LOG="${LOG_DIR}/${SLURM_JOBID}.err"
CONFIG=${LOG_DIR}/config.json

# copy batch script for posterity
cp -v /var/spool/slurmd/job$(printf %05u $SLURM_JOB_ID)/slurm_script ${LOG_DIR}/batch.sh

echo ""
echo "--------------------------------------------------------------------------------------------------"
echo "--------------------------------------------------------------------------------------------------"
echo "{" | tee -a ${CONFIG}
echo "\"NUM_NODES\"             : \"${SLURM_NNODES}\"," | tee -a ${CONFIG}
echo "\"TOTAL_GPUS\"            : \"${TOTAL_GPUS}\"," | tee -a ${CONFIG}
echo "\"GPUS_PER_NODE\"         : \"${GPUS_PER_NODE}\"," | tee -a ${CONFIG}
echo "\"POWER_LIMIT\"           : \"${POWER_LIMIT}\"," | tee -a ${CONFIG}
echo "\"NODES\"                 : \"$(scontrol show hostnames | tr '\n' ',')\"," | tee -a ${CONFIG}
echo "\"LOG_DIR\"               : \"${LOG_DIR}\"," | tee -a ${CONFIG}
echo "\"ERR_LOG\"               : \"${ERR_LOG}\"," | tee -a ${CONFIG}
echo "\"CONFIG\"                : \"${CONFIG}\"," | tee -a ${CONFIG}
echo "\"OUTPUT_DIR\"            : \"${OUTPUT_DIR}\"," | tee -a ${CONFIG}
echo "\"START_TIME\"            : \"${START_TIME}\"," | tee -a ${CONFIG}
echo ""
echo "--------------------------------------------------------------------------------------------------"
echo ""
echo "\"MODEL_TYPE\"            : \"${MODEL_TYPE}\"," | tee -a ${CONFIG}
echo "\"TOKENIZER_NAME\"        : \"${TOKENIZER_NAME}\"," | tee -a ${CONFIG}
echo "\"DATASET_NAME\"          : \"${DATASET_NAME}\"," | tee -a ${CONFIG}
echo "\"DATASET_CONFIG\"        : \"${DATASET_CONFIG}\"," | tee -a ${CONFIG}
echo "\"BATCH_SIZE\"            : \"${BATCH_SIZE}\"," | tee -a ${CONFIG}
echo "\"NUM_EPOCHS\"            : \"${NUM_EPOCHS}\"," | tee -a ${CONFIG}
echo "--------------------------------------------------------------------------------------------------"
echo "--------------------------------------------------------------------------------------------------"
echo ""

# # set power limit
# echo "setting power limit"
# sidhelper ${POWER_LIMIT}
# 
# # setup NVIDIA monitoring
# echo "starting GPU monitoring for job ${SLURM_JOBID}"
# dcgmi group -c allgpus --default
# dcgmi stats -g 2 -e
# dcgmi stats -g 2 -s $SLURM_JOBID -v
# export DCGM_LOG=${LOG_DIR}/dcgm.log

# # These lines will setup nvidia-smi monitoring. We don't monitor jobs on tx-green, so we have to do this ourselves.
# echo "nvidia-smi file : ${TMPDIR}/${SLURM_JOBID}-$(uname -n)-nvidia-smi.csv"
# CVD=$(tr -dc '[:alnum:]-,' <<< $CUDA_VISIBLE_DEVICES)
# (${HOME}/bin/nvidia-smi-profile.sh ${CVD} ${TMPDIR}/${SLURM_JOBID}-$(uname -n)-nvidia-smi.csv & )
# PID=$(pidof nvidia-smi)
# echo "PID of profile script : ${PID}"

srun --ntasks-per-node=2 run_mlm_prolog.sh ${POWER_LIMIT}

mpirun ${MPI_FLAGS} -x MASTER_ADDR -x MASTER_PORT -x HF_HOME -x TRANSFORMERS_OFFLINE -x HF_DATASETS_OFFLINE -x WANDB_DISABLED run_mlm.py \
    --model_type ${MODEL_TYPE} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --output_dir ${OUTPUT_DIR} 2>${ERR_LOG} 1>${LOG_FILE}

srun --ntasks-per-node=2 run_mlm_epilog.sh ${LOG_DIR}

# # stop collection and show stats
# echo "stopping NVIDIA DCGM stat collection for job ${SLURM_JOBID}"
# dcgmi stats -x $SLURM_JOBID -v
# echo "writing stats to file : ${DCGM_LOG}"
# echo "NVIDIA DCGM stats for job ${SLURM_JOBID} : "
# dcgmi stats -j $SLURM_JOBID -v 2>&1 > ${DCGM_LOG}

# # stop nvidia-smi logging
# kill -9 ${PID}
# mv -v ${TMPDIR}/${SLURM_JOBID}-$(uname -n)-nvidia-smi.csv ${LOG_DIR}/

echo "=> end of script"

WALL_TIME=$(sacct --format="ElapsedRaw" -j ${SLURM_JOBID} -n | head -n1 | awk '{$1=$1};1')
END_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo "\"END_TIME\"              : \"${END_TIME}\"," | tee -a ${CONFIG}
echo "\"WALL_TIME\"             : \"${WALL_TIME}\"" | tee -a ${CONFIG}
echo "}" | tee -a ${CONFIG}

echo "all done"
