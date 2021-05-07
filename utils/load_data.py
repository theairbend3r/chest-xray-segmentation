import pathlib

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, SubsetRandomSampler


class DatasetChestXrayMontShznJsrt(Dataset):
    def __init__(
        self,
        images_path,
        masks_path,
        image_transform,
        img_msk_size,
        dataset_name,
        dsubset="full",
    ):
        super(DatasetChestXrayMontShznJsrt, self).__init__()

        self.images_path = pathlib.Path(images_path)
        self.masks_path = pathlib.Path(masks_path)

        self.dsubset = dsubset

        self.img_msk_size = img_msk_size

        self.transform = image_transform

        if dataset_name in ["shenzhen", "montgomery"]:
            if self.dsubset == "full":
                self.img_file_paths = [f for f in images_path.glob("*.png")]
                self.msk_file_paths = [f for f in masks_path.glob("*.png")]
            elif self.dsubset == "normal":
                self.img_file_paths = [f for f in images_path.glob("*0.png")]
                self.msk_file_paths = [f for f in masks_path.glob("*0.png")]
            elif self.dsubset == "abnormal":
                self.img_file_paths = [f for f in images_path.glob("*1.png")]
                self.msk_file_paths = [f for f in masks_path.glob("*1.png")]
            else:
                raise ValueError("Invalid dsubset value.")

        elif dataset_name == "jsrt":
            if self.dsubset == "full":
                self.img_file_paths = [f for f in images_path.glob("*.png")]
                self.msk_file_paths = [f for f in masks_path.glob("*.png")]
            elif self.dsubset == "normal":
                self.img_file_paths = [f for f in images_path.glob("*.png")]
                self.img_file_paths = [
                    f for f in self.img_file_paths if f.stem[3] == "N"
                ]
                self.msk_file_paths = [f for f in masks_path.glob("*.png")]
                self.msk_file_paths = [
                    f for f in self.msk_file_paths if f.stem[3] == "N"
                ]
            elif self.dsubset == "abnormal":
                self.img_file_paths = [f for f in images_path.glob("*.png")]
                self.img_file_paths = [
                    f for f in self.img_file_paths if f.stem[3] == "L"
                ]
                self.msk_file_paths = [f for f in masks_path.glob("*.png")]
                self.msk_file_paths = [
                    f for f in self.msk_file_paths if f.stem[3] == "L"
                ]
            else:
                raise ValueError("Invalid dsubset value.")
        else:
            raise ValueError("Invalid dataset_name")

    def __getitem__(self, idx):
        img_path = str(self.img_file_paths[idx])
        msk_path = str(self.msk_file_paths[idx])

        img = (
            Image.open(img_path)
            .convert("L")
            .resize((self.img_msk_size, self.img_msk_size), Image.BICUBIC)
        )

        img = self.transform(img)

        msk = (
            Image.open(msk_path)
            .convert("L")
            .resize((self.img_msk_size, self.img_msk_size), Image.NEAREST)
        )

        # Convert pixels of value=255 to 1 so that it plays well with Pytorch
        msk = np.array(msk)
        msk[msk == 255] = 1
        msk = torch.from_numpy(msk).long()

        return img, msk

    def __len__(self):
        assert len(self.img_file_paths) == len(self.msk_file_paths)
        return len(self.img_file_paths)


IMAGE_MASK_SIZE = 512


def create_samplers(dataset, train_percent):
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))

    np.random.shuffle(dataset_indices)

    train_split_index = int(np.floor(train_percent * dataset_size))

    train_idx = dataset_indices[:train_split_index]
    val_idx = dataset_indices[train_split_index:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    return train_sampler, val_sampler
