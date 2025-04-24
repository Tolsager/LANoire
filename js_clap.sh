#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -J clap_ee
#BSUB -W 2:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -u s204141@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo outputs/clap.out
#BSUB -eo outputs/clap.err
uv run train_clap_ee.py