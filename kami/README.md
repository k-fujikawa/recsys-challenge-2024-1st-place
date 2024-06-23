# RecSys Challenge 2024 (kami634 part)

## Structure
```text
.
├── Dockerfile.cpu
├── README.md
├── compose.cpu.yaml
├── exp
├── input
├── output
├── utils
└── yamls
```

## 1. Machine Specifications
Google Compute Engine: n2-highmem-96
- **Operating System**: [Debian GNU/Linux 11]
- **CPU**: [Intel(R) Xeon(R) CPU @ 2.80GHz, 96 vCPUs]
- **RAM**: [768 GB]
- **Storage**: [500GB Standard persistent disk & Google Cloud Storage] 


## 2. Directory&Data Preparation
1. Create the `input`&`output` directory if it doesn't exist:
    ```bash
    mkdir -p input
    mkdir -p output
    ```
2. Download the dataset `ebnerd_large` and `ebnerd_testset` from the competition website.
    - If you want to use debug mode, download `ebnerd_small`.
3. Move the downloaded dataset to the `input` directory:


## 3. Environment Setup
1. **Start Docker and enter bash**:
    - It mounts input&output directory
    ```sh
    docker compose -f compose.cpu.yaml build
    docker compose -f compose.cpu.yaml run --rm kaggle bash 
    ```

## 4. Training and Inference
**Note**: This process requires significant time and memory.

If you want to use debug mode, append `--debug` option

1. Create Candidates
    ```sh
    inv create-candidates
    ```

2. Feature Extraction
    ```sh
    inv create-features
    ```

3. Create Datasets
    ```sh
    inv create-datasets
    ```

4. Train & Inference
    ```sh
    inv train
    ```