# RecSys Challenge 2024 (kfujikawa part)

## 1. Machine Specifications

I mainly used Google Compute Engine: c2d-highmem-112

- OS: Ubuntu 20.04.2
- CPU:  [AMD EPYC 7B13, 112 vCPUs]
- RAM: 896 GB
- Python: 3.12.1

## 2. Setup

```bash
pip install -U poetry
poetry install
```

## 3. Ensembling
**Note**: This process requires significant time and memory.

if you want to use debug option, append `--debug` option.

```bash
poetry run python src/weighted_mean.py --debug
poetry run python src/stacking.py --debug
poetry run python src/make_submission.py
```
