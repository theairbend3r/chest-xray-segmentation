# Chest X-Ray Segmentation

This repository contains PyTorch code train segmentation models on chest-xray images.

## Directory Structure

```
.
├── data
├── train_seg.py
└── utils
    ├── load_data.py
    ├── load_extra.py
    ├── load_loss.py
    ├── load_model.py
    └── load_train_loop.py
```

## Specifications

### Datasets

Dataloaders for following datasets are available. Images should be saved in `./data`.

1. Montgomery
2. Shenzhen
3. JSRT

### Dataset Modes

Segmentation can be run on the following configurations.

1. Full: Runs on complete dataset.
2. Normal: Runs only on normal chest x-rays.
3. Abnormal: Runs only on abnormal chest x-rays.

### Segmentation Models

The following segmentation models are available. Please note that these are custom implementations and don't exeactly match the actual papers.

1. FCN8
2. DeconvNet

## Run

### Command Blueprint

```
python train_seg.py --dataset [str] --subset [str]  --model_name [str] --batch_size [int] --epochs [int] --learning_rate [float]
```

### Example Command

```
python train_seg.py --dataset "montgomery" --subset "full"  --model_name "custom_model_name" --batch_size 16 --epochs 100 --learning_rate 0.01
```
