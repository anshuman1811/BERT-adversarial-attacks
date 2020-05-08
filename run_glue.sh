#!/bin/bash

#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00
#SBATCH --mem=60000

MODEL_NAME=roberta-base
TASK=MRPC
MODEL_DIR=/mnt/nfs/work1/696ds-s20/purujitgoyal/$TASK
GLUE_DIR=/mnt/nfs/scratch1/purujitgoyal/adversarial_glue_data/$TASK
OUTPUT_DIR=/mnt/nfs/work1/696ds-s20/purujitgoyal

# ~/anaconda3/envs/685/bin/python ../transformers/examples/run_glue.py --model_type roberta --model_name $MODEL_NAME --task_name $TASK --do_train --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir $OUTPUT_DIR/$TASK
# ~/anaconda3/envs/685/bin/python ../transformers/examples/run_glue.py --model_type roberta --model_name $MODEL_NAME --task_name $TASK --do_train --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK --max_seq_length 256 --per_gpu_train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir $OUTPUT_DIR/$TASK
~/anaconda3/envs/685/bin/python ../transformers/examples/run_glue.py --model_type roberta --model_name_or_path $MODEL_DIR --task_name $TASK --do_eval --do_lower_case --data_dir $GLUE_DIR --max_seq_length 256 --per_gpu_train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir $MODEL_DIR
