from pathlib import Path


ROOT_DIR = Path(__file__).parents[2].resolve()
DATA_DIR = ROOT_DIR / "data"
KFUJIKAWA_DIR = DATA_DIR / "kfujikawa"
PREPROCESS_DIR = KFUJIKAWA_DIR / "v0xxx_preprocess"
RAWDATA_DIRS = {
    "articles": DATA_DIR / "ebnerd" / "articles.parquet",
    "train": DATA_DIR / "ebnerd" / "train",
    "validation": DATA_DIR / "ebnerd" / "validation",
    "test": DATA_DIR / "ebnerd" / "ebnerd_testset" / "test",
}
TRAIN_START_UNIXTIME = 1684360800
SPLIT_INTERVAL = 60 * 60 * 24 * 7
