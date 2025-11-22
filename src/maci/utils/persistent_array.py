import numpy as np
from typing import Any
from pathlib import Path

class PersistentArray:
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
        
        filename: Path = Path(filename)
        if not filename.exists() or not filename.is_file():
            filename.parent.mkdir(parents=True, exist_ok=True)
            filename.touch(exist_ok=False)
            np.save(filename, np.zeros(shape=(max_length, item_size), dtype=dtype))

        self._array = np.memmap(filename, shape=(max_length, item_size), dtype=dtype, mode="r+")
    

    def push(self, item: np.ndarray):
        self._array[self._end, :] = item
        self._end = (self._end + 1) % self._max_length
        self._length = min(self._length + 1, self._max_length)


    def flush(self) -> None:
        self._array.flush()
    

    def sample(self) -> np.ndarray:
        return self._rng.choice(self._array[:self._length], min(self._length, self._sample_size))
    

    def restarted(self) -> bool:
        return self._end == 0
    

    def reset(self) -> None:
        self._end = 0
        self._length = 0


    def get(self, i) -> Any:
        return self._array[i]