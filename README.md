# Hate Speech Targets Multi-Label Classification

This repository contains experimental code and resources to train and evaluate language models for hate speech multi-label classification task. The aim of the project is to investigate the performance of different models, mainly BERT-based, on the task of hate speech classification.

The dataset provided in this repository contains 90k samples from various sources and created by combining multiple existing datasets (see `datasets/dataset-preprocessed.csv`). The dataset was preprocessed and cleaned to ensure consistency and quality of the data. The dataset contains text samples and corresponding labels for hate speech.

Original datasets used for creating the unified dataset:

| Dataset Name      | Source/Author                                                                                         | Github                                                                      |
| ----------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Multitarget CONAN | https://aclanthology.org/2021.acl-long.250/                                                           | https://github.com/marcoguerini/CONAN                                       |
| MLMA              | https://aclanthology.org/D19-1474/                                                                    | https://github.com/HKUST-KnowComp/MLMA_hate_speech                          |
| MMHS150K          | https://arxiv.org/abs/1910.03814                                                                      | -                                                                           |
| Waseem & Hoovy    | https://aclanthology.org/W16-5618/                                                                    | https://github.com/zeeraktalat/hatespeech                                   |
| Waseem            | https://aclanthology.org/N16-2013/                                                                    | https://github.com/zeeraktalat/hatespeech                                   |
| DataSociety       | https://www.kaggle.com/datasets/crowdflower/twitter-user-gender-classification                        | -                                                                           |
| Twitter Vidhya    | https://www.analyticsvidhya.com/blog/2021/06/twitter-sentiment-analysis-a-nlp-use-case-for-beginners/ | https://github.com/sachink382/Twitter-Sentiment-Analysis---Analytics-Vidhya |

## Installation & Usage

**Prerequisites:**

- Python >= 3.8
- CUDA capable GPU (NVIDIA) and [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Ubuntu 20.04 - project may also run on other distros with some minor tweaking of package versions

**Installation:**

Install required packages from `requirements.txt` (installation may vary depending on the environment):

```bash
pip install -r requirements.txt
```

Install following torch packages:

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Add executable permission to training script:

```bash
chmod +x train_in_bg.sh
```

## Project structure

We strongly recommend to read [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://www.pytorchlightning.ai/) documentations
for better understanding of project and file structure as well as built-in functions used in this project.

- `notebooks` - contains Jupyter notebooks used for data exploration, visualization and processing. Its describes the creation of the unified dataset by combining multiple sources.

- `benchmark` - contains dataset and model checkpoints used for performing benchmark.
  Benchmark can be run using `benchmark.py` file in root directory.

- `datasets` - contains datasets used for preprocessing and training:

  - `dataset_preprocessed.csv` - cleaned and preprocessed dataset used for training
  - `unified_dataset.csv` - raw dataset created from other datasets - original dataset names are descibed in column `original_dataset`

- `models` - model classes (containing model architecture and parameters) and training functions for `baseline` and `bert_based` models

- `utils` - helper functions and utilities

- `constants.py` - configuration file containing all the hyper-parameters and constants used in the project

- `preprocess.py` - preprocessing and logging (version control) of datasets

- `test_sample_text.py` - outputs prediction for a given sentence using given model checkpoints

- `train_in_bg.sh` - helper script which calls `main.py` and runs training of model in background (also outputs process id in `save_pid.txt` and all the logs in the `training_output.log`)

## Usage

Before training, you may want to change which model architecture is going to be trained and change parameters.
To do so, you need to import given model in either
`/models/baselines/train.py` or `/models/bert_based/train.py`, change import to desired train function inside `main.py` and tweak parameters according to your needs directly in model file (e.g. removing/adding layers in `Model_LSTM.py`) or through available parameters as arguments.

Example of running training with default settings (baseline `Model_LSTM.py`):

```bash
$ ./train_in_bg

> Enter model name: example_model_name_123
```

Afterwards, training output and results will be printed inside `training_output.log`.

## Experiment tracking

This project uses [Wandb](https://wandb.ai/site) for experiment tracking. You can enable wandb by changing `WANDB` in `constants.py` to `True`.
