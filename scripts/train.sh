#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

mkdir -p logs
PATH_TO_DATA=$GLUE_PATH

MODEL_TYPE=${1}
MODEL_SIZE=${2}
DATASET=${3}
ROUTINE=${4}
LAMB=${5}

LR=2e-5
EPOCHS=3
MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
if [ $MODEL_TYPE = 'bert' ]
then
  MODEL_NAME=${MODEL_NAME}-uncased
fi
if [ $MODEL_TYPE = 'albert' ]
then
  MODEL_NAME=${MODEL_NAME}-v2
fi

echo ${MODEL_TYPE}-${MODEL_SIZE}/$DATASET $ROUTINE
python -um examples.run_glue \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME \
  --task_name $DATASET \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $PATH_TO_DATA/$DATASET \
  --max_seq_length 128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=256 \
  --learning_rate $LR \
  --num_train_epochs $EPOCHS \
  --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/${ROUTINE} \
  --overwrite_output_dir \
  --plot_data_dir ./plotting/ \
  --overwrite_cache \
  --mc_dropout \
  --top2_diff \
  --train_routine $ROUTINE \
  --lamb $LAMB

if [ $DATASET = 'MNLI' ] || [ $DATASET = 'bMNLI' ]
then
  scripts/eval.sh ${MODEL_TYPE} ${MODEL_SIZE} ${DATASET}-mm $ROUTINE $LAMB
fi
