#!/bin/bash -ex

EXP_DIR=$(cd $(dirname $0)/..; pwd)

# Preparation for pseudo labeling
poetry run python $EXP_DIR/v8xxx_ensemble/v8004_015_016_v1170_v1174.py

# Training for pseudo labeling
DEVICE=0
MODEL=$EXP_DIR/v1xxx_training/v1184_111_PL_bert_L4_256d.py
poetry run python $MODEL train common.fold=0 trainer.devices=[${DEVICE}] common.overwrite=True
poetry run python $MODEL train common.fold=2 trainer.devices=[${DEVICE}] common.overwrite=True
poetry run python $MODEL predict --split valid common.fold=0 trainer.devices=[${DEVICE}]
poetry run python $MODEL predict --split test common.fold=2 trainer.devices=[${DEVICE}]
