import numpy as np
from typing import Any
from pathlib import Path

class ReplayBuffer:
    def __init__(
            self,
            filename: str,
            item_size: int,
            max_length: int,
            dtype: type,
            sample_size: int,
            rng: np.random.Generator
        ) -> None:
        self._max_length = max_length
        self._length = self._end = 0
        self._sample_size = sample_size
        self._rng = rng
        
        file_path = Path(filename)
        if not file_path.exists() or not file_path.is_file():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch(exist_ok=False)
            np.save(file_path, np.zeros(shape=(max_length, item_size), dtype=dtype))

        self._array = np.memmap(filename, shape=(max_length, item_size), dtype=dtype, mode="r+")
    

    def push(self, item: np.ndarray):
        self._array[self._end, :] = item
        self._end = (self._end + 1) % self._max_length
        self._length = min(self._length + 1, self._max_length)


    def flush(self) -> None:
        self._array.flush()
    

    def sample(self) -> np.ndarray:
        return self._rng.choice(self._array[:self._length], min(self._length, self._sample_size))