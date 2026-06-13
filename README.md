# HLinear

This repository provides the implementation of the paper **From Values to Oscillation States: Hilbert-Guided Forecasting for Multivariate Time Series**.

We propose **HLinear**, a lightweight multivariate time series forecasting model guided by Hilbert oscillation priors. HLinear extracts explicit oscillation-state information from raw time series via the Hilbert transform and introduces a Hilbert-guided Channel Encoder (HGCE) to enhance the modeling of hidden dynamic states and inter-variable oscillation coupling. The goal is to achieve a favorable balance among forecasting accuracy, computational efficiency, and memory usage.

## 1. Highlights

* **Hilbert oscillation priors**: Extract envelope- and phase-related information through the Hilbert transform.
* **Oscillation Prior Extraction (OPE)**: Generate compact variable-wise oscillation priors.
* **Oscillation Channel Encoding (OCE)**: Inject oscillation priors into the channel encoding process for oscillation-aware cross-variable interaction.
* **Lightweight forecasting**: Maintain low computational complexity, memory usage, and inference overhead.
* **Reproducible experiments**: Provide source code, dataset download instructions, and experimental configurations.

## 2. Repository Structure

```text
HLinear/
├── configs/
│   └── HLinear_configs.png
├── data_provider/
│   ├── data_factory.py
│   └── data_loader.py
├── exp/
│   ├── exp_basic.py
│   └── exp_main.py
├── layers/
│   ├── Embed.py
│   ├── RevIN.py
│   └── Transformer_EncDec.py
├── models/
│   └── HLinear.py
├── utils/
├── README.md
├── requirements.txt
└── run.py
```

The core implementation of HLinear is provided in `models/HLinear.py`.
The `data_provider/` folder contains data loading and preprocessing scripts.
The `exp/` folder contains the training and evaluation pipeline.
The entry point for running experiments is `run.py`.

## 3. Environment Setup

We recommend creating an independent Conda environment:

```bash
conda create -n hlinear python=3.8
conda activate hlinear
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Please install the appropriate PyTorch version according to your local CUDA environment.

## 4. Data Preparation

You can obtain the well-preprocessed datasets from Google Drive:

```text
https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2
```

After downloading the datasets, please place them in the folder `../dataset`, for example:

```text
../dataset/
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
├── ETTm2.csv
├── weather.csv
├── electricity.csv
├── traffic.csv
└── ...
```

## 5. Running Example

Experiments can be launched through `run.py`. A basic example is shown below:

```bash
python run.py \
  --model HLinear \
  --data ETTh1 \
  --data_path ETTh1.csv \
  --seq_len 96 \
  --pred_len 96 \
  --d_model 512 \
  --batch_size 32 \
  --learning_rate 0.0001
```

For more experimental settings and model configurations, please refer to `run.py` and `configs/HLinear_configs.png`.

## 6. Reproducibility

To reproduce the main experimental results reported in the paper, please make sure that:

1. The datasets are downloaded and placed in the `../dataset` directory.
2. The running environment is consistent with `requirements.txt`.
3. The training parameters, prediction lengths, and dataset settings follow the experimental settings in the paper.
4. The experiments are launched through the parameter interface provided in `run.py`.

## 7. Citation

This manuscript is currently under review. The official citation information will be updated after the paper is accepted or publicly available.

If you would like to cite this work temporarily, please use the following format:

```bibtex
@misc{li2026hlinear,
  title        = {From Values to Oscillation States: Hilbert-Guided Forecasting for Multivariate Time Series},
  author       = {Li, Ziqiong and Chai, Heyu and Yue, Wenzhen and Liu, Xinru and Liu, Shengjun},
  year         = {2026},
  note         = {Manuscript under review}
}
```

## 8. Contact

If you have any questions, please feel free to open an issue or contact the author.

* Ziqiong Li
* Central South University
* Email: [242101007@csu.edu.cn](mailto:242101007@csu.edu.cn)
