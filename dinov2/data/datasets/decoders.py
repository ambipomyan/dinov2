# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from io import BytesIO
from typing import Any

from PIL import Image

from numpy import ndarray

import pickle

class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError


class ImageDataDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    #def decode(self) -> Image:
    #    f = BytesIO(self._image_data)
    #    return Image.open(f).convert(mode="RGB")
    def decode(self, idx) -> ndarray:
        f = BytesIO(self._image_data)
        full_img = pickle.load(f)
        return full_img[idx]


class TargetDecoder(Decoder):
    def __init__(self, target: Any):
        self._target = target

    def decode(self) -> Any:
        return self._target
