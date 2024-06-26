#!/bin/bash -ex

EXP_DIR=$(cd $(dirname $0)/..; pwd)

# Base datasets
MODELS=(
    # $EXP_DIR/v1xxx_training/v1157_111_fix_past_v2.py
    $EXP_DIR/v1xxx_training/v1170_111_L8_128d.py
    $EXP_DIR/v1xxx_training/v1174_111_L8_128d_smpl3_drophist.py
)
DEVICE=0

for MODEL in ${MODELS[@]}; do
    poetry run python $MODEL train common.fold=0 trainer.devices=[${DEVICE}] common.overwrite=True
    poetry run python $MODEL train common.fold=2 trainer.devices=[${DEVICE}] common.overwrite=True
    poetry run python $MODEL predict --split validation common.fold=0 trainer.devices=[${DEVICE}]
    poetry run python $MODEL predict --split test common.fold=2 trainer.devices=[${DEVICE}]
done
