#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=$GLUE_PATH

MODEL_TYPE=${1}
MODEL_SIZE=${2}
DATASET=${3}
ROUTINE=${4}
HP=${5}

if [ $ROUTINE = 'multi-stage' ]
then
  CONF_TH=$HP
  PROP=$HP
  # only one of the above two will be used,
  # depending on whether random_multi_stage_proportion is used
  LAMB=0.1
else
  LAMB=$HP
  CONF_TH=1.0
fi

MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
OUTPUT_DIR="./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/${ROUTINE}"
if [ $MODEL_TYPE = 'bert' ]
then
  MODEL_NAME=${MODEL_NAME}-uncased
fi
if [ $MODEL_TYPE = 'albert' ]
then
  MODEL_NAME=${MODEL_NAME}-v2
fi
if [ $DATASET = 'MNLI-mm' ]
then
  OUTPUT_DIR="./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/MNLI/${ROUTINE}"
fi
if [ $DATASET = 'bMNLI-mm' ]
then
  OUTPUT_DIR="./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/bMNLI/${ROUTINE}"
fi

echo ${MODEL_TYPE}-${MODEL_SIZE}/$DATASET $ROUTINE
python -um examples.run_glue \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME \
  --task_name $DATASET \
  --do_eval \
  --do_lower_case \
  --data_dir $PATH_TO_DATA/$DATASET \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size=512 \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --plot_data_dir ./plotting/ \
  --overwrite_cache \
  --mc_dropout \
  --top2_diff \
  --train_routine $ROUTINE \
  --lamb $LAMB \
  --conf_th $CONF_TH \
  --multi_stage_base raw


# for mc_dropout evaluation of different dropout_prob and repetitive_runs
# change mc_dropout to multi_mc_dropout, and specify dropout_prob

# for multi_stage evaluation, use the multi-stage routine
# change multi_stage_base to raw/reg-hist
# add --random_multi_stage_proportion $PROP if necessary (use raw as multi_stage_base)
