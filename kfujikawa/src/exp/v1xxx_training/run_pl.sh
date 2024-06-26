#!/bin/bash -ex

EXP_DIR=$(cd $(dirname $0)/..; pwd)
DEVICE=0
MODEL=$EXP_DIR/v1xxx_training/v1184_111_PL_bert_L4_256d.py
poetry run python $MODEL train common.fold=0 trainer.devices=[${DEVICE}] common.overwrite=True
poetry run python $MODEL train common.fold=2 trainer.devices=[${DEVICE}] common.overwrite=True
poetry run python $MODEL predict --split validation common.fold=0 trainer.devices=[${DEVICE}]
poetry run python $MODEL predict --split test common.fold=2 trainer.devices=[${DEVICE}]
