from mmap import mmap
from pathlib import Path
import pickle
from typing import TypeVar

T = TypeVar("T")


class MMapObject:
    def __init__(self, obj: T, filename: str, resize_factor: float=1.5) -> None:
        file_path = Path(filename)
        file_existed = file_path.exists()
        self._file = open(file_path, "rb+" if file_existed else "wb+")
        self._mmap = mmap(self._file.fileno(), 0)

        if not file_existed:
            self._file.truncate(8)
            self._mmap.resize(8)
            self.unload_object(obj)

        self._resize_factor = resize_factor


    def load_object(self) -> T:
        obj_size = int.from_bytes(self._mmap[:8])
        return pickle.loads(self._mmap[8:obj_size])


    def unload_object(self, obj: T) -> None:
        data = pickle.dumps(obj)
        self._mmap[:8] = len(data).to_bytes(8)

        if self._mmap.size() < 8 + len(data):
            new_size = max(8 + len(data), int(self._mmap.size() * self._resize_factor))
            self._file.truncate(new_size)
            self._mmap.resize(new_size)

        self._mmap[8:] = data


    def flush(self):
        self._mmap.flush()