#!/usr/bin/env python
# coding: utf-8
# Author:  Akshaj Verma


import os
import pathlib
import argparse

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader


## Util Imports
from utils.load_data import DatasetChestXrayMontShznJsrt, create_samplers
from utils.load_model import ModelDeconvSmall
from utils.load_train_loop import train_loop


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(100)


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, help="Dataset")
parser.add_argument("--subset", type=str, help="Subset")
parser.add_argument("--model_name", type=str, help="Model Name")
parser.add_argument("--img_msk_size", type=int, help="Image and Mask Size")
parser.add_argument("--batch_size", type=int, help="Batch Size")
parser.add_argument("--epochs", type=int, help="Epochs")
parser.add_argument("--learning_rate", type=float, help="Learning Rate")

ap = parser.parse_args()

######################
##      Config      ##
######################

DATA_NAME = ap.dataset  # montgomery, shenzhen, jsrt
DATA_SUBSET = ap.subset  # full, normal, abnormal
MODEL_NAME = ap.model_name  # "deconv_small_decodertransposeinit_sqex"

IMAGE_MASK_SIZE = ap.img_msk_size  # 512
BATCH_SIZE = ap.batch_size  # 16
EPOCHS = ap.epochs  # 60
LEARNING_RATE = ap.learning_rate  # 0.01


print(f"\nDataset Name: {DATA_NAME}")
print(f"Data Subset: {DATA_SUBSET}")
print(f"Model Name: {MODEL_NAME}")

print(f"Input size = {IMAGE_MASK_SIZE}")
print(f"Batch size = {BATCH_SIZE}")
print(f"Epochs = {EPOCHS}")
print(f"Learning Rate= {LEARNING_RATE}")

# Summary Writer
writer = SummaryWriter(f"runs/{DATA_NAME}_{DATA_SUBSET}_{MODEL_NAME}")


## Define Paths and Set GPU
images_path = pathlib.Path(f"../../data/{DATA_NAME}/train/images/")
masks_path = pathlib.Path(f"../../data/{DATA_NAME}/train/masks/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nWe're using =>", device)


######################
##    Load Data     ##
######################


## Create Dataset
image_transforms = {
    "train": transforms.Compose([transforms.ToTensor()]),
    "test": transforms.Compose([transforms.ToTensor()]),
}

print("\nImages path: ", images_path)
print("Masks path: ", masks_path)

# Initialize Dataset
dataset = DatasetChestXrayMontShznJsrt(
    images_path=images_path,
    masks_path=masks_path,
    image_transform=image_transforms["train"],
    img_msk_size=IMAGE_MASK_SIZE,
    dataset_name=DATA_NAME,
    dsubset=DATA_SUBSET,
)


## Train - Val Split
train_sampler, val_sampler = create_samplers(dataset, 0.8)


## Create Dataloaders
train_loader = DataLoader(
    dataset=dataset, shuffle=False, batch_size=BATCH_SIZE, sampler=train_sampler,
)

val_loader = DataLoader(
    dataset=dataset, shuffle=False, batch_size=1, sampler=val_sampler
)


## Data Sanity Check
print(f"\nTrain loader = {next(iter(train_loader))[0].shape}")
print(f"Val loader = {next(iter(val_loader))[0].shape}")
print(f"Train loader length = {len(train_loader)}")
print(f"Val loader length = {len(val_loader)}")


#################################
##    Load Model and Config    ##
#################################

# init model
model = ModelDeconvSmall(num_class=2)

# Sample data loader examples
x_train_example, y_train_example = next(iter(train_loader))
y_pred_example = model(x_train_example)
print("\nShape of output mask = ", y_pred_example.shape)

# Add model graph to tensorboard
writer.add_graph(model, (next(iter(train_loader))[0]))

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005
)

scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001)


model_save_path = f"../../models/seg"
print(f"\nModels are saved here: {model_save_path}")

# Move model to GPU (if available)
model.to(device)
print("\n")
print("=" * 50)

######################
##    Train Loop    ##
######################

print("\nBegin training.")

train_loop(
    epochs=EPOCHS,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    writer=writer,
    data_name=DATA_NAME,
    data_subset=DATA_SUBSET,
    model_name=MODEL_NAME,
    train_loader=train_loader,
    val_loader=val_loader,
    model_save_path=model_save_path,
    device=device,
)

# close tensorboard writer
writer.close()

print("\n\nTraining is over.")
