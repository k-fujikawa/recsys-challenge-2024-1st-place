# RecSys Challenge 2024

1st place solution by team :D for RecSys Challenge 2024.

## 1. kami part

Change directory to `kami` and train the models according to the [README](./kami/README.md) instructions.  
The prediction parquet files will be output to the following paths.

```
./kami/output/experiments/015_train_third/large067_001/validation_result_first.parquet
./kami/output/experiments/015_train_third/large067_001/test_result_third.parquet
./kami/output/experiments/016_catboost/large067/validation_result_first.parquet
./kami/output/experiments/016_catboost/large067/test_result_third.parquet
```


## 2. kfujikawa part

Change directory to `kfujikawa` and train the models according to the [README](./kfujikawa/README.md) instructions.  
The prediction parquet files will be output to the following paths.

```
./kfujikawa/data/kfujikawa/v1xxx_training/v1157_111_fix_past_v2/fold_0/predictions/validation.parquet
./kfujikawa/data/kfujikawa/v1xxx_training/v1157_111_fix_past_v2/fold_2/predictions/test.parquet
./kfujikawa/data/kfujikawa/v1xxx_training/v1170_111_L8_128d/fold_0/predictions/validation.parquet
./kfujikawa/data/kfujikawa/v1xxx_training/v1170_111_L8_128d/fold_2/predictions/test.parquet
./kfujikawa/data/kfujikawa/v1xxx_training/v1184_111_PL_bert_L4_256d/fold_0/predictions/validation.parquet
./kfujikawa/data/kfujikawa/v1xxx_training/v1184_111_PL_bert_L4_256d/fold_2/predictions/test.parquet
```


## 3. sugawarya part

Change directory to `sugawarya` and train the models according to the [README](./sugawarya/README.md) instructions.
The prediction parquet and submission files will be output to the following paths.

```
./sugawarya/output/test_weighted_mean.parquet
./sugawarya/output/test_stacking.parquet
./sugawarya/output/v999_final_submission.zip
```
