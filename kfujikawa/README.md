# RecSys Challenge 2024 (kfujikawa part)

## 1. Setup

### 1.1. Experiment environment

- OS: Ubuntu 20.04.2
- GPU: NVIDIA L4
- NVIDIA Driver: 555.42.02
- CUDA: 12.5
- Python: 3.10.10

### 1.2. Install python dependencies

```bash
pip install -U poetry
poetry install
```

## 2. Feature extraction

```bash
./src/exp/v0xxx_preprocess/run.sh
```

## 3. Training

### 3.1. Scratch training

```bash
./src/exp/v1xxx_training/run_scratch.sh
```

### 3.2. Pseudo labeling training

```bash
./src/exp/v1xxx_training/run_pl.sh
```
