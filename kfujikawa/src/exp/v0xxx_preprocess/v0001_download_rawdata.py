#! /usr/bin/env python

import requests
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import typer
from loguru import logger
from pathlib import Path
from tqdm.auto import tqdm

from exputils.const import DATA_DIR

APP = typer.Typer(pretty_exceptions_enable=False)
URLS = [
    # "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/articles_large_only.zip",
    # "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
    "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip",
    "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip",
    "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip",
    "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/predictions_large_random.zip",
    # "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip",
    # "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_image_embeddings.zip",
    # "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip",
    # "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip",
    # "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/FacebookAI_xlm_roberta_base.zip",
]


def _download_from_url(url: str, dest_path: Path, file_num: int):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1KB
    progress_bar = tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        position=file_num,
        desc=f"Download: {url}",
        leave=False,
    )
    temp_path = dest_path.with_name(dest_path.name + ".tmp")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "wb") as temp_file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            temp_file.write(data)
    progress_bar.close()
    temp_path.rename(dest_path)


def _download_files(output_dir: Path, overwrite: bool = False):
    with ThreadPoolExecutor() as executor:
        file_num = 0
        futures = []
        for url in URLS:
            parsed_url = urlparse(url)
            output_path = output_dir / parsed_url.path[1:]
            if overwrite:
                output_path.unlink(missing_ok=True)
            if output_path.exists():
                continue
            futures.append(
                executor.submit(
                    _download_from_url,
                    url=url,
                    dest_path=output_path,
                    file_num=file_num + 1,
                )
            )
            file_num += 1
        [
            future.result()
            for future in tqdm(
                as_completed(futures),
                position=0,
                desc="Download files",
                total=len(futures),
            )
        ]


def _unzip_file(zip_file: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dest_dir)


def _unzip_files(input_dir: Path, output_dir: Path):
    for zip_path in input_dir.glob("**/*.zip"):
        _output_dir = output_dir
        if zip_path.stem == "ebnerd_small":
            _output_dir = output_dir.with_name("ebnerd_small")
        _output_dir = _output_dir / zip_path.relative_to(input_dir).parent.name
        logger.info(f"Unzipping {zip_path.name} to {_output_dir}")
        _unzip_file(zip_path, _output_dir)


@APP.command()
def download(
    output_dir: Path = typer.Option(
        DATA_DIR / "ebnerd_archives",
        help="Path to the input directory",
    ),
    overwrite: bool = typer.Option(
        False,
        help="Whether to overwrite the existing files",
    ),
):
    _download_files(
        output_dir=output_dir,
        overwrite=overwrite,
    )


@APP.command()
def unzip(
    input_dir: Path = typer.Option(
        DATA_DIR / "ebnerd_archives",
        help="Path to the input directory",
    ),
    output_dir: Path = typer.Option(
        DATA_DIR / "ebnerd",
        help="Path to the input directory",
    ),
    overwrite: bool = typer.Option(
        False,
        help="Whether to overwrite the existing files",
    ),
):
    if output_dir.exists() and not overwrite:
        logger.info(f"{output_dir} already exists. Skipping.")
        return

    _unzip_files(
        input_dir=input_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    APP()
