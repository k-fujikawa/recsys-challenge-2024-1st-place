# RecSys Challenge 2024 (kami634 part)

## Structure
```text
.
├── Dockerfile.cpu
├── README.md
├── compose.cpu.yaml
├── exp
├── input
├── notebook
├── output
├── utils
└── yamls: データのパスなど各スクリプトに共通する設定を管理
```

## 1. Machine Specifications
Google Compute Engine: n2-highmem-64
- **Operating System**: [Debian GNU/Linux 11]
- **CPU**: [Intel(R) Xeon(R) CPU @ 2.80GHz, 64 vCPUs]
- **RAM**: [512 GB]
- **Storage**: [500GB Standard persistent disk & Google Cloud Storage] 


## 2. Data Preparation
1. Create the `input` directory if it doesn't exist:
    ```bash
    mkdir -p input
    ```
2. Download the dataset `ebnerd_large` from the competition website.
    - If you want to debug, download `ebnerd_small`.
3. Move the downloaded dataset to the `input` directory:


## 3. Environment Setup
1. Create the `output` directory if it doesn't exist:
    ```bash
    mkdir -p output
    ```
2. **Start Docker and enter bash**:
    - It mounts input&output directory
    ```sh
    docker compose -f compose.cpu.yaml build
    docker compose -f compose.cpu.yaml run --rm kaggle bash 
    ```

## 4. Feature extraction
**Note**: This process requires significant time and memory.

```sh
# demo用テストデータの抽出
python preprocess/test_demo/run.py 
```
