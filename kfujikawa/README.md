# RecSys Challenge 2024 (kfujikawa part)

## 1. Machine Specifications

I mainly used Google Compute Engine: g2-standard-32

- OS: Ubuntu 20.04.2
- GPU: NVIDIA L4 x 1
- NVIDIA Driver: 555.42.02
- CUDA: 12.5
- Python: 3.10.10

## 2. Setup

```bash
pip install -U poetry
poetry install
```

## 3. Feature extraction

```bash
./src/exp/v0xxx_preprocess/run.sh
```

## 4. Training

### 4.1. Scratch training

```bash
./src/exp/v1xxx_training/run_scratch.sh
```

### 4.2. Create Pseudo labels

```bash
./src/exp/v8xxx_ensemble/run_ensemble.sh
```

### 4.3. Pseudo labeling training

```bash
./src/exp/v1xxx_training/run_pl.sh
```
