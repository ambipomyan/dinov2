'''
most of the code is copied from https://github.com/ambipomyan/dinov2/blob/main/dinov2/data/datasets/image_net.py
'''

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset

import pickle

from tifffile import imread, imwrite
import pandas as pd

from myutils import get_keys_from_txt

logger = logging.getLogger("dinov2")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
        #    _Split.TRAIN: 1_281_167,
        #    _Split.VAL: 50_000,
        #    _Split.TEST: 100_000,
            #_Split.TRAIN: 149_425, # 63_425 + 86_000
            #_Split.VAL: 63_425,
            #_Split.TEST: 63_425,
            _Split.TRAIN: 2_155_046, # 298_464 + 166_456 + ...
            _Split.VAL: 298_464,
            _Split.TEST: 362_818, # 166456 + 196362
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
        dirname = self.get_dirname(class_id)
        #if self == _Split.TRAIN:
        #    basename = f"{class_id}_{actual_index}"
        #else:  # self in (_Split.VAL, _Split.TEST):
            #basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        #return os.path.join(dirname, basename + ".JPEG")
        basename = f"{class_id}_{actual_index}"

        return os.path.join(dirname, basename + "")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        #class_id = os.path.split(dirname)[-1]
        class_id = filename.split("_")[0] + "_" + filename.split("_")[1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index

class HEDataset(ExtendedVisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "HEDataset.Split",
        root: str,
        extra: str,
        seg: str,
        names: str,
        picks: str,
        size: int,
        adds: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        self._entries = None
        self._class_ids = None
        self._class_names = None

##################
# load tiff file #
##################
        ## keys
        keys = []
        full_channels = os.listdir(names)
        picked_channels = os.listdir(picks)
# order
        full_channels.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
        picked_channels.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))

        for c in range(len(full_channels)):
            full = os.path.join(names, full_channels[c])
            picked = os.path.join(picks, picked_channels[c])
            key = get_keys_from_txt(full, picked)
            keys.append(key)
        ## seg
        self.seg = seg
        ## window
        w_x = int(size//2) # cast to int
        w_y = int(size//2)
        ## csvs
        xs = []
        ys = []
        curr_seg = seg
        curr_csv_names = os.listdir(curr_seg)
# order
        curr_csv_names.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))

        for csv_name in curr_csv_names:
            curr_full_csv_name = os.path.join(curr_seg, csv_name)
            df = pd.read_csv(curr_full_csv_name, usecols=['X_centroid','Y_centroid'])
            x = df['X_centroid'].to_numpy().astype(int)
            y = df['Y_centroid'].to_numpy().astype(int)
            xs.append(x)
            ys.append(y)
        ## imgs
        imgs = []
        curr_root = os.path.join(root, self._split.value)
        curr_file_names = os.listdir(curr_root)
# order
        curr_file_names.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
        #print(curr_file_names)

        for i, file_name in enumerate(curr_file_names):
            curr_full_file_name = os.path.join(curr_root, file_name)
            with open(curr_full_file_name, 'rb') as f:
                tmp = imread(f, key=keys[i])
                page = tmp[0]
                height, width = page.shape
                ## get indexes
                for index in range(len(xs[i])):
                    ori_x = xs[i][index]
                    ori_y = ys[i][index]
                    upper = max(ori_y - w_y, 0)
                    lower = min(ori_y + w_y, height)
                    left  = max(ori_x - w_x, 0)
                    right = min(ori_x + w_x, width)
                    ## make sure that each of the inputs are of the same size
                    if upper == 0:
                        lower = 2 * w_y
                    if lower == height:
                        upper = height - 2*w_y
                    if left == 0:
                        right = 2 * w_x
                    if right == width:
                        left = width - 2*w_x
                    ## get images
                    img = tmp[:, upper:lower, left:right]
                    imgs.append(img)

        if not(adds):
            self.imgs = imgs
        else:
################
# load HE file #
################
            ## imgs_adds
            imgs_adds = []
            curr_adds = os.path.join(adds, self._split.value)
            curr_adds_file_names = os.listdir(curr_adds)
# order
            curr_adds_file_names.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
            #print(curr_adds_file_names)

            for i, file_name in enumerate(curr_adds_file_names):
                curr_full_file_name = os.path.join(curr_adds, file_name)
                with open(curr_full_file_name, 'rb') as f:
                    tmp = imread(f)
                    page = tmp[0]
                    height, width = page.shape
                    ## get indexes
                    for index in range(len(xs[i])):
                        ori_x = xs[i][index]
                        ori_y = ys[i][index]
                        upper = max(ori_y - w_y, 0)
                        lower = min(ori_y + w_y, height)
                        left  = max(ori_x - w_x, 0)
                        right = min(ori_x + w_x, width)
                        ## make sure that each of the inputs are of the same size
                        if upper == 0:
                            lower = 2 * w_y
                        if lower == height:
                            upper = height - 2*w_y
                        if left == 0:
                            right = 2 * w_x
                        if right == width:
                            left = width - 2*w_x
                        ## get images
                        img = tmp[:, upper:lower, left:right]
                        imgs_adds.append(img)
#########
# merge #
#########
            #print(2*w_x)
            #print(len(imgs_adds))
            #print(len(imgs))

            self.imgs = np.concatenate((imgs_adds, imgs), axis=1)

    @property
    def split(self) -> "HEDataset.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _class_ids_path(self) -> str:
        return f"class-ids-{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        return f"class-names-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        if self._split == _Split.TEST:
            assert False, "Class IDs are not available in TEST split"
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        assert self._class_ids is not None
        return self._class_ids

    def _get_class_names(self) -> np.ndarray:
        if self._split == _Split.TEST:
            assert False, "Class names are not available in TEST split"
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        assert self._class_names is not None
        return self._class_names

    def find_class_id(self, class_index: int) -> str:
        class_ids = self._get_class_ids()
        return str(class_ids[class_index])

    def find_class_name(self, class_index: int) -> str:
        class_names = self._get_class_names()
        return str(class_names[class_index])

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        actual_index = entries[index]["actual_index"]

        class_id = self.get_class_id(index)

        image_relpath = self.split.get_image_relpath(actual_index, class_id)
        image_full_path = os.path.join(self.root, image_relpath)
        #with open(image_full_path, mode="rb") as f:
        #    image_data = f.read()
        image_data = self.imgs[index]

        return image_data

    def get_target(self, index: int) -> Optional[Target]:
        entries = self._get_entries()
        class_index = entries[index]["class_index"]
        return None if self.split == _Split.TEST else int(class_index)

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return None if self.split == _Split.TEST else entries["class_index"]

    def get_class_id(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_id = entries[index]["class_id"]
        return None if self.split == _Split.TEST else str(class_id)

    def get_class_name(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_name = entries[index]["class_name"]
        return None if self.split == _Split.TEST else str(class_name)

    def __len__(self) -> int:
        entries = self._get_entries()
        assert len(entries) == self.split.length
        return len(entries)

    def _load_labels(self, labels_path: str) -> List[Tuple[str, str]]:
        labels_full_path = os.path.join(self.root, labels_path)
        labels = []

        try:
            with open(labels_full_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    class_id, class_name = row
                    labels.append((class_id, class_name))
        except OSError as e:
            raise RuntimeError(f'can not read labels file "{labels_full_path}"') from e

        return labels

    def _dump_entries(self) -> None:
        split = self.split
        if split == HEDataset.Split.TEST:
            dataset = None
            sample_count = split.length
            max_class_id_length, max_class_name_length = 0, 0
        else:
            labels_path = "labels.txt"
            logger.info(f'loading labels from "{labels_path}"')
            labels = self._load_labels(labels_path)

            # NOTE: Using torchvision ImageFolder for consistency
            #from torchvision.datasets import ImageFolder
            from mydataset import MyImageFolder

            dataset_root = os.path.join(self.root, split.get_dirname())
            #dataset = ImageFolder(dataset_root)
            dataset = MyImageFolder(dataset_root, self.seg)
            sample_count = len(dataset)
            max_class_id_length, max_class_name_length = -1, -1
            for sample in dataset.samples:
                _, class_index = sample
                class_id, class_name = labels[class_index]
                max_class_id_length = max(len(class_id), max_class_id_length)
                max_class_name_length = max(len(class_name), max_class_name_length)

        dtype = np.dtype(
            [
                ("actual_index", "<u4"),
                ("class_index", "<u4"),
                ("class_id", f"U{max_class_id_length}"),
                ("class_name", f"U{max_class_name_length}"),
            ]
        )
        entries_array = np.empty(sample_count, dtype=dtype)

        if split == HEDataset.Split.TEST:
            old_percent = -1
            for index in range(sample_count):
                percent = 100 * (index + 1) // sample_count
                if percent > old_percent:
                    logger.info(f"creating entries: {percent}%")
                    old_percent = percent

                actual_index = index + 1
                class_index = np.uint32(-1)
                class_id, class_name = "", ""
                entries_array[index] = (actual_index, class_index, class_id, class_name)
        else:
            class_names = {class_id: class_name for class_id, class_name in labels}

            assert dataset
            old_percent = -1
            for index in range(sample_count):
                percent = 100 * (index + 1) // sample_count
                if percent > old_percent:
                    logger.info(f"creating entries: {percent}%")
                    old_percent = percent

                image_full_path, class_index = dataset.samples[index]
                image_relpath = os.path.relpath(image_full_path, self.root)
                class_id, actual_index = split.parse_image_relpath(image_relpath)
                class_name = class_names[class_id]
                entries_array[index] = (actual_index, class_index, class_id, class_name)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def _dump_class_ids_and_names(self) -> None:
        split = self.split
        if split == HEDataset.Split.TEST:
            return

        entries_array = self._load_extra(self._entries_path)

        max_class_id_length, max_class_name_length, max_class_index = -1, -1, -1
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            max_class_index = max(int(class_index), max_class_index)
            max_class_id_length = max(len(str(class_id)), max_class_id_length)
            max_class_name_length = max(len(str(class_name)), max_class_name_length)

        class_count = max_class_index + 1
        class_ids_array = np.empty(class_count, dtype=f"U{max_class_id_length}")
        class_names_array = np.empty(class_count, dtype=f"U{max_class_name_length}")
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            class_ids_array[class_index] = class_id
            class_names_array[class_index] = class_name

        logger.info(f'saving class IDs to "{self._class_ids_path}"')
        self._save_extra(class_ids_array, self._class_ids_path)

        logger.info(f'saving class names to "{self._class_names_path}"')
        self._save_extra(class_names_array, self._class_names_path)

    def dump_extra(self) -> None:
        self._dump_entries()
        self._dump_class_ids_and_names()
