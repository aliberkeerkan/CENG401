"""In-memory image store for the prototype.

Keeps the most recent uploads in RAM. Data is lost when the server
restarts; a later version can move this to disk or Redis without
changing the API.
"""

import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

MAX_IMAGES = 50


@dataclass
class StoredImage:
    rgb: np.ndarray
    alpha: np.ndarray | None


class ImageStore:
    def __init__(self, capacity: int = MAX_IMAGES):
        self._items: OrderedDict[str, StoredImage] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def add(self, rgb: np.ndarray, alpha: np.ndarray | None) -> str:
        image_id = "img_" + secrets.token_hex(4)
        with self._lock:
            self._items[image_id] = StoredImage(rgb, alpha)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)  # drop the oldest
        return image_id

    def get(self, image_id: str) -> StoredImage | None:
        with self._lock:
            item = self._items.get(image_id)
            if item is not None:
                self._items.move_to_end(image_id)
            return item


store = ImageStore()
