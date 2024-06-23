#!/bin/bash -ex

FILE_DIR=$(cd $(dirname $0); pwd)
OPTIONS="--skip"
# OPTIONS="--overwrite"

# Base datasets
poetry run python $FILE_DIR/v0001_download_rawdata.py download
poetry run python $FILE_DIR/v0001_download_rawdata.py unzip

# Base datasets
poetry run python $FILE_DIR/v0100_articles.py $OPTIONS
poetry run python $FILE_DIR/v0200_users.py $OPTIONS
poetry run python $FILE_DIR/v0300_impressions.py $OPTIONS

# Additional features
poetry run python $FILE_DIR/v0101_article_inviews_in_split.py $OPTIONS
poetry run python $FILE_DIR/v0101_article_inviews_in_split_v2.py $OPTIONS
poetry run python $FILE_DIR/v0102_article_metadata_id_v2.py $OPTIONS
poetry run python $FILE_DIR/v0103_article_history_counts.py $OPTIONS
poetry run python $FILE_DIR/v0201_user_inviews_in_split.py $OPTIONS
poetry run python $FILE_DIR/v0301_imp_counts_per_user.py $OPTIONS
