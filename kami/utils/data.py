from pathlib import Path


def get_data_dirs(input_dir, size_name="large"):
    return {
        "train": input_dir / f"ebnerd_{size_name}" / "train",
        "validation": input_dir / f"ebnerd_{size_name}" / "validation",
        "test": input_dir / "ebnerd_testset" / "ebnerd_testset" / "test"
        if size_name == "large"
        else Path("/kaggle/working/output/preprocess/test_demo/base/test"),
    }
