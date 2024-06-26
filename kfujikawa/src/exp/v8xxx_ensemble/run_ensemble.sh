#!/bin/bash -ex

EXP_DIR=$(cd $(dirname $0)/..; pwd)
poetry run python $EXP_DIR/v8xxx_ensemble/v8004_015_016_v1170_v1174.py valid
poetry run python $EXP_DIR/v8xxx_ensemble/v8004_015_016_v1170_v1174.py test
