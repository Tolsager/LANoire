#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -J roberta_ee
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
##BSUB -u s204141@student.dtu.dk
##BSUB -B
##BSUB -N
#BSUB -oo outputs/roberta.out
#BSUB -eo outputs/roberta.err
uv run train_trans_ee.py
