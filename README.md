# Transformers for Selective Prediction

This is the code base for the paper The Art of Abstention: Selective Prediction and Error Regularization for Natural Language Processing.

## Installation

This repo is tested with Python 3.7, PyTorch 1.3.1, and Cuda 10.1. Using a virtulaenv or conda environemnt is recommended, for example:

```
conda install pytorch==1.3.1 torchvision cudatoolkit=10.1 -c pytorch
```

After installing the required environment, clone this repo, and install the following requirements:

```
git clone https://github.com/castorini/transformers-selective
cd transformers-selective
pip install -r ./requirements.txt
conda install scikit-learn
```

## Preparing data

Most datasets used (MRPC, QNLI, MNLI) are from GLUE. Check [this](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) for more details. After preparing GLUE, specify the environment variable $GLUE_PATH to GLUE's directory.

### For running the no-answer problem part (Section 5.4 in the paper)

SST-5 has to be downloaded separately. After downloading it, put it in $GLUE_PATH as if it's also a part of GLUE.

There are 2 scripts in `data_preprocess` to preprocess SST-5 and binarize MNLI and SST-5. They have to be moved to corresponding directory and executed there (check the scripts for details).

## Training and Evaluation

There are two scripts in the `scripts` folder, which can be run from the repo root, e.g., `scripts/train.sh`.


In each script, there are several things to modify before running:

* path to the GLUE dataset ($GLUE_PATH).
* path for saving fine-tuned models. Default: `./saved_models`.
* path for saving evaluation results. Default: `./plotting`. Results are printed to stdout and also saved to `npy` files in this directory to facilitate plotting figures and further analyses.
* model_type (`lstm`, `bert`, or `albert`)
* model_size (`base` or `large` for `bert`; `base` only for others)
* dataset (MRPC, QNLI, MNLI, bMNLI, bSST-5)
* routine: what regularization to use (`raw`, `reg-curr`, and `reg-hist`)
* lamb: the regularization hyperparameter

#### train.sh

This is for training/fine-tuning AND evaluating models.

#### eval.sh

This is for evaluating a trained model. Check the script for further details (e.g., how to obtain data for Figures 3 and 4 in the paper).



## Citation

Please cite our paper if you find the repository useful.

